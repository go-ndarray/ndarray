package kernels

import "runtime"

// Mat-vec and dot kernels. The packed GEMM that used to share a file with them
// lives in github.com/go-blas/blas; these need no packing.

// matVecThreshold is the m*k element count above which MatVecP fans the row-dots
// across cores. Mat-vec is a single bandwidth-bound streaming pass over A (one
// multiply-add per element, no reuse), so its parallel crossover is far higher
// than the compute-bound GEMM's: forking it on a few-thousand-element A is a net
// loss (100x100 mat-vec measures ~2.5us serial vs ~5.8us parallel). It is a var
// so tests can pin it.
var matVecThreshold = 1 << 14

// dotRange returns sum(a[i]*b[i]) over equal-length slices, using four
// independent accumulators, so four multiply-add chains are in flight instead of
// one (gc does not vectorise this loop; the gain is latency hiding). The
// four-way grouping is a fixed
// reassociation of the sum; MatVecP/Dot1DP document the ULP trade-off where it is
// observable. It is the contiguous building block for both mat·vec and the 1-D
// dot — neither needs to go through the packing GEMM.
func dotRange(a, b []float64) float64 {
	var s0, s1, s2, s3 float64
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// MatVecP computes dst = a(m x k) * v(k,) as m independent contiguous row-dots,
// fanned across cores above matVecThreshold. Each output dst[i] is the dot of
// row i of a (the contiguous slice a[i*k:(i+1)*k]) with v, so the access pattern
// is pure unit-stride streaming — far cheaper than routing a single-column GEMM
// through the packer. dst need not be pre-zeroed (each entry is written, not
// accumulated). Rows are disjoint, so the parallel result equals the serial one.
func MatVecP(dst, a, v []float64, m, k int) { MatVecStridedP(dst, a, k, v, m, k) }

// MatVecStridedP is MatVecP on rows lda elements apart: row i of a is
// a[i*lda : i*lda+k], so a sub-block of a larger row-major matrix is
// multiplied in place.
func MatVecStridedP(dst, a []float64, lda int, v []float64, m, k int) {
	body := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = dotSIMD(a[i*lda:i*lda+k], v)
		}
	}
	if m*k < matVecThreshold {
		body(0, m)
		return
	}
	w := runtime.GOMAXPROCS(0)
	if w > m {
		w = m
	}
	parallelFor(m, w, body)
}

// VecMatP computes dst = v(k,) * a(k x n) = column sums weighted by v, i.e.
// dst[j] = sum_p v[p]*a[p*n+j]. It accumulates row by row so the inner loop over
// j streams a contiguous a row and the contiguous dst, which is the cache-
// friendly traversal (the transpose-free analogue of mat·vec for the 1-D·2-D
// case). dst must be zeroed by the caller.
func VecMatP(dst, v, a []float64, k, n int) {
	for p := 0; p < k; p++ {
		vp := v[p]
		row := a[p*n : p*n+n]
		for j := 0; j < n; j++ {
			dst[j] += vp * row[j]
		}
	}
}

// Dot1DP returns the inner product of two equal-length contiguous vectors,
// fanned across cores above ParThreshold: each chunk is dot-producted with the
// SIMD kernel and the partials are summed. The chunked four-way
// grouping reassociates the sum (a few ULP, like SumP / numpy pairwise); callers
// needing the exact left-to-right value use the scalar Dot1D.
func Dot1DP(a, b []float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return dotSIMD(a, b)
	}
	w := numWorkers(n)
	parts := min(4*w, max(w, n/reduceGrain)) // see mapReduceP
	partials := make([]float64, parts)
	chunk := (n + parts - 1) / parts
	parallelFor(parts, w, func(lo, hi int) {
		for idx := lo; idx < hi; idx++ {
			s := idx * chunk
			if s >= n {
				// chunk rounding can leave trailing workers past the end (e.g.
				// many workers over a short, threshold-forced input); their
				// partial is just the additive identity.
				partials[idx] = 0
				continue
			}
			e := s + chunk
			if e > n {
				e = n
			}
			partials[idx] = dotSIMD(a[s:e], b[s:e])
		}
	})
	return Sum(partials)
}
