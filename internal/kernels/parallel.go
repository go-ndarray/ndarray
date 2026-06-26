package kernels

import (
	"runtime"
	"sync"
)

// Multicore kernels.
//
// NumPy's elementwise and reduction ufuncs are single-threaded SIMD C, so the
// durable way a pure-Go library beats them on large arrays is to spread the
// work across CPU cores. These parallel kernels partition the contiguous
// element range into one chunk per worker and run the SAME scalar kernels (the
// oracle, above) on each chunk, so correctness is inherited unchanged and the
// result is independent of the worker count. A size threshold keeps small
// arrays on the serial path, where goroutine scheduling would dominate.

// ParThreshold is the minimum element count at which the parallel kernels fan
// out across goroutines. Below it the serial scalar kernel is used, because the
// goroutine launch/join cost exceeds the work. It is a var so tests can force
// the parallel path on small inputs (and pin it for deterministic coverage).
var ParThreshold = 32768

// numWorkers returns the worker count for n elements: at most GOMAXPROCS (which
// is always >= 1), and never more than one worker per chunk-sized slab so a
// tiny-but-above-threshold input does not spawn maximum goroutines for a few
// elements each. Callers only reach it with n >= ParThreshold >= 1, so the slab
// bound is itself >= 1.
func numWorkers(n int) int {
	w := runtime.GOMAXPROCS(0)
	if maxByWork := (n + ParThreshold - 1) / ParThreshold; maxByWork < w {
		w = maxByWork
	}
	return w
}

// parallelFor splits [0,n) into w contiguous, near-equal blocks and runs body
// on each block in its own goroutine, waiting for all to finish. body must be
// safe to call concurrently on disjoint [lo,hi) ranges.
func parallelFor(n, w int, body func(lo, hi int)) {
	if w <= 1 {
		body(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// binaryKernel is the shape of the elementwise scalar kernels (Add, Mul, …).
type binaryKernel func(dst, a, b []float64)

// AddP, SubP, MulP, DivP are the multicore elementwise kernels. Each runs the
// matching scalar kernel on disjoint chunks of the (already broadcast/
// contiguous) operands when the length crosses ParThreshold, and serially
// otherwise. Output is identical to the serial kernel for every input.
func runBinaryP(k binaryKernel, dst, a, b []float64) {
	n := len(dst)
	if n < ParThreshold {
		k(dst, a, b)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) {
		k(dst[lo:hi], a[lo:hi], b[lo:hi])
	})
}

// AddP, SubP, MulP, DivP run the SIMD-vectorized elementwise kernel (addBin etc:
// packed SSE2 on amd64, NEON D2 on arm64, scalar oracle on the other four)
// serially below ParThreshold and fanned across cores above it. The SIMD kernel
// is bit-identical to the scalar oracle (each lane is an exact, independent
// IEEE-754 op — no reduction grouping), so the result matches the serial scalar
// computation for every input.

// AddP writes a[i]+b[i] into dst[i], parallelised above ParThreshold.
func AddP(dst, a, b []float64) { runBinaryP(addBin, dst, a, b) }

// SubP writes a[i]-b[i] into dst[i], parallelised above ParThreshold.
func SubP(dst, a, b []float64) { runBinaryP(subBin, dst, a, b) }

// MulP writes a[i]*b[i] into dst[i], parallelised above ParThreshold.
func MulP(dst, a, b []float64) { runBinaryP(mulBin, dst, a, b) }

// DivP writes a[i]/b[i] into dst[i], parallelised above ParThreshold.
func DivP(dst, a, b []float64) { runBinaryP(divBin, dst, a, b) }

// MapP applies f to every element of src into dst, parallelised above
// ParThreshold. f must be safe for concurrent calls (the math ufuncs are).
func MapP(dst, src []float64, f func(float64) float64) {
	n := len(dst)
	if n < ParThreshold {
		Map(dst, src, f)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) {
		Map(dst[lo:hi], src[lo:hi], f)
	})
}

// SqrtP writes sqrt(src[i]) into dst[i], routed through the SIMD sqrt kernel
// (packed SQRTPD on amd64; the intrinsic-lowered scalar FSQRTD loop on
// arm64/others) and parallelised above ParThreshold. Unlike MapP it does NOT
// pass math.Sqrt through a func-pointer — that indirection blocks the compiler's
// FSQRTD intrinsic and the packed SSE2 kernel, the very thing that made Sqrt
// lose to numpy. The result is bit-identical to a scalar math.Sqrt loop.
func SqrtP(dst, src []float64) {
	n := len(dst)
	if n < ParThreshold {
		sqrtSIMD(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) {
		sqrtSIMD(dst[lo:hi], src[lo:hi])
	})
}

// mapReduceP computes one partial per worker by applying red to that worker's
// contiguous chunk, then folds the partials with combine. The grouping it
// imposes is a valid (and, for the associative reducers, identical) reordering
// of the serial reduction.
//
// chunk rounds up, so with w workers the chunk-sized stride can overrun n before
// the last worker is reached: a worker's start s = idx*chunk can land at or past
// n (this happens when GOMAXPROCS is large relative to the element count — many
// more workers than ParThreshold-sized slabs). Such a worker owns no elements;
// it must NOT index a[s:e] (s >= n would panic) and must NOT contribute a partial
// (there is no reduction identity that is bit-identical for Max/Min). We compute
// the count of workers that actually own a chunk and produce exactly that many
// partials, so the fold sees only real data and the result is unchanged.
func mapReduceP(a []float64, red func([]float64) float64, combine func([]float64) float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return red(a)
	}
	w := numWorkers(n)
	chunk := (n + w - 1) / w
	// active = number of workers whose start s = idx*chunk is < n; the rest own
	// nothing and are dropped (their start would overrun n and panic a[s:e], and
	// there is no bit-identical reduction identity to feed combine for Max/Min).
	// Since chunk = ceil(n/w) >= n/w, active = ceil(n/chunk) is always <= w, so
	// dropping trailing workers never loses real data.
	active := (n + chunk - 1) / chunk
	partials := make([]float64, active)
	parallelFor(active, active, func(lo, hi int) {
		for idx := lo; idx < hi; idx++ {
			s := idx * chunk
			e := s + chunk
			if e > n {
				e = n
			}
			partials[idx] = red(a[s:e])
		}
	})
	return combine(partials)
}

// SumP returns the sum of all elements of a, computed as the SIMD-summed
// per-chunk partials folded by a final Sum. Floating-point addition is not
// associative, so this grouping can differ from the strictly-sequential Sum by
// a few ULP — the same trade-off numpy's pairwise summation makes; callers that
// need the exact sequential value use Sum.
func SumP(a []float64) float64 { return mapReduceP(a, sumSIMD, Sum) }

// reduceP folds per-chunk results of the associative scalar reducer red with
// red itself, giving a result identical to the serial kernel.
func reduceP(a []float64, red func([]float64) float64) float64 {
	return mapReduceP(a, red, red)
}

// axisKernel is the shape of the [outer][axisLen][inner] axis reducers
// (SumAxis, MaxAxis, …).
type axisKernel func(dst, src []float64, outer, axisLen, inner int)

// RunAxisP runs an axis-reduction kernel, fanning the `outer` slabs across cores
// above ParThreshold (measured by the total element count outer*axisLen*inner).
// Each worker owns a disjoint band of outer rows and writes the matching disjoint
// [band*inner] region of dst, reading only its own src slab — so the parallel
// result is identical to the serial kernel. Splitting `outer` keeps every
// worker's inner traversal contiguous (the cache-friendly axis), which is the
// case that matters: an axis-1 reduction of an (R x C) matrix has outer=R,
// inner=1, and the serial kernel leaves all but one core idle.
func RunAxisP(k axisKernel, dst, src []float64, outer, axisLen, inner int) {
	total := outer * axisLen * inner
	if total < ParThreshold {
		k(dst, src, outer, axisLen, inner)
		return
	}
	// Split the outer slabs: each worker owns a disjoint band of outer rows and
	// writes the matching disjoint dst rows, reading only its own contiguous src
	// slab — identical to the serial kernel. This is the case that matters most:
	// an axis-1 reduction of an (R x C) matrix has outer=R, inner=1, so the serial
	// kernel leaves all but one core idle, and an inner-split would need a strided
	// column gather whose copy cost cancels the parallel gain. A single-outer-slab
	// reduction (axis 0: outer=1, inner=C) already streams contiguous inner runs
	// per accumulation step and stays on the (fast) serial path.
	if outer < 2 {
		k(dst, src, outer, axisLen, inner)
		return
	}
	w := numWorkers(total)
	if w > outer {
		w = outer
	}
	parallelFor(outer, w, func(lo, hi int) {
		dstSlab := dst[lo*inner : hi*inner]
		srcSlab := src[lo*axisLen*inner : hi*axisLen*inner]
		k(dstSlab, srcSlab, hi-lo, axisLen, inner)
	})
}

// MaxP returns the maximum element of a (non-empty), parallelised above
// ParThreshold. It runs the SIMD max kernel (packed MAXPD + NaN scan on amd64;
// the intrinsic-lowered scalar FMAXD reducer on arm64/others) on each chunk and
// folds the partials with the same NaN-propagating kernel. Max is associative
// and NaN-propagating, so the result is exactly the serial Max — including
// returning NaN whenever any element is NaN (numpy.max semantics).
func MaxP(a []float64) float64 { return reduceP(a, maxSIMD) }

// MinP returns the minimum element of a (non-empty), parallelised above
// ParThreshold, via the SIMD min kernel; result identical to the serial Min,
// NaN-propagating like numpy.min.
func MinP(a []float64) float64 { return reduceP(a, minSIMD) }
