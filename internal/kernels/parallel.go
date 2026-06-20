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

// AddP writes a[i]+b[i] into dst[i], parallelised above ParThreshold.
func AddP(dst, a, b []float64) { runBinaryP(Add, dst, a, b) }

// SubP writes a[i]-b[i] into dst[i], parallelised above ParThreshold.
func SubP(dst, a, b []float64) { runBinaryP(Sub, dst, a, b) }

// MulP writes a[i]*b[i] into dst[i], parallelised above ParThreshold.
func MulP(dst, a, b []float64) { runBinaryP(Mul, dst, a, b) }

// DivP writes a[i]/b[i] into dst[i], parallelised above ParThreshold.
func DivP(dst, a, b []float64) { runBinaryP(Div, dst, a, b) }

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
// contiguous chunk, then folds the partials with combine. numWorkers caps the
// worker count at one per ParThreshold-sized slab, so every chunk is non-empty
// and exactly w partials are produced. The grouping it imposes is a valid (and,
// for the associative reducers, identical) reordering of the serial reduction.
func mapReduceP(a []float64, red func([]float64) float64, combine func([]float64) float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return red(a)
	}
	w := numWorkers(n)
	partials := make([]float64, w)
	chunk := (n + w - 1) / w
	parallelFor(w, w, func(lo, hi int) {
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

// GemmThreshold is the minimum number of result elements (m*n) at which MatMulP
// fans the GEMM across goroutines. Small products stay single-threaded.
var GemmThreshold = 1 << 14

// MatMulP computes dst = a(m x k) * b(k x n) like MatMul, but splits the m
// output rows across cores above GemmThreshold. Each goroutine owns a disjoint
// band of rows, so the per-row MatMul kernel writes non-overlapping regions of
// dst — the result is identical to the serial MatMul, just faster. dst must be
// zeroed by the caller.
func MatMulP(dst, a, b []float64, m, k, n int) {
	if m*n < GemmThreshold {
		MatMul(dst, a, b, m, k, n)
		return
	}
	// One worker per core, never more than one per output row. parallelFor then
	// splits the m rows into w contiguous bands; each band's MatMul writes a
	// disjoint slice of dst (rows [r0,r1)), so there is no sharing on the output
	// and b is read-only — the result is identical to the serial MatMul.
	w := runtime.GOMAXPROCS(0)
	if w > m {
		w = m
	}
	parallelFor(m, w, func(r0, r1 int) {
		MatMul(dst[r0*n:r1*n], a[r0*k:r1*k], b, r1-r0, k, n)
	})
}
