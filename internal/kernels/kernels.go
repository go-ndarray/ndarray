// Package kernels holds the inner numeric loops for the ndarray package.
//
// Every loop here is a portable, pure-Go (CGO=0) scalar kernel. They are kept
// behind this small, contiguous-slice API so that SIMD variants can drop in
// later without changing callers or tests: Phase 1 of the roadmap replaces
// these with go-asmgen-generated kernels across all six 64-bit Go SIMD targets
// (amd64, arm64, riscv64, loong64, ppc64le, s390x), selected at runtime, while
// these scalar versions remain the reference and the fallback.
//
// Kernels operate on flat, contiguous []float64 slices. Shape, stride and
// broadcasting concerns live in the parent package; by the time a slice reaches
// a kernel it is already materialised in row-major order.
package kernels

import "math"

// Add writes a[i]+b[i] into dst[i] for every element.
func Add(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// Sub writes a[i]-b[i] into dst[i] for every element.
func Sub(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// Mul writes a[i]*b[i] into dst[i] for every element.
func Mul(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

// Div writes a[i]/b[i] into dst[i] for every element.
func Div(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

// Map applies f to every element of src, writing the result into dst.
func Map(dst, src []float64, f func(float64) float64) {
	for i := range dst {
		dst[i] = f(src[i])
	}
}

// sqrtScalar is the portable square-root oracle: dst[i] = sqrt(src[i]) for every
// element, following Go's math.Sqrt (and IEEE-754) exactly — including
// sqrt(-x)=NaN, sqrt(-0)=-0, sqrt(+Inf)=+Inf, sqrt(NaN)=NaN. The packed SIMD
// kernels (amd64 SQRTPD, arm64 FSQRT) compute the same correctly-rounded IEEE
// square root lane-by-lane, so they are bit-identical to this oracle (sqrt,
// unlike a sum reduction, is a single rounded operation with no grouping
// freedom). It is the dispatch fallback on the non-vectorized arches and the
// reference the per-arch CI validates the .s against bit-for-bit.
func sqrtScalar(dst, src []float64) {
	for i := range dst {
		dst[i] = math.Sqrt(src[i])
	}
}

// b2f maps a boolean comparison result to a 0/1 float mask value.
func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// Equal writes the a[i]==b[i] mask (1/0) into dst[i].
func Equal(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] == b[i])
	}
}

// NotEqual writes the a[i]!=b[i] mask (1/0) into dst[i].
func NotEqual(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] != b[i])
	}
}

// Greater writes the a[i]>b[i] mask (1/0) into dst[i].
func Greater(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] > b[i])
	}
}

// GreaterEqual writes the a[i]>=b[i] mask (1/0) into dst[i].
func GreaterEqual(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] >= b[i])
	}
}

// Less writes the a[i]<b[i] mask (1/0) into dst[i].
func Less(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] < b[i])
	}
}

// LessEqual writes the a[i]<=b[i] mask (1/0) into dst[i].
func LessEqual(dst, a, b []float64) {
	for i := range dst {
		dst[i] = b2f(a[i] <= b[i])
	}
}

// Maximum writes the pairwise maximum of a[i] and b[i] into dst[i].
func Maximum(dst, a, b []float64) {
	for i := range dst {
		dst[i] = math.Max(a[i], b[i])
	}
}

// Minimum writes the pairwise minimum of a[i] and b[i] into dst[i].
func Minimum(dst, a, b []float64) {
	for i := range dst {
		dst[i] = math.Min(a[i], b[i])
	}
}

// Sum returns the sum of all elements of a.
func Sum(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v
	}
	return s
}

// Prod returns the product of all elements of a.
func Prod(a []float64) float64 {
	p := 1.0
	for _, v := range a {
		p *= v
	}
	return p
}

// Max returns the maximum element of a, which must be non-empty.
//
// NaN convention: Max is NaN-propagating — if any element is NaN the result is
// NaN. This matches numpy.max (and numpy.maximum), Go's builtin max, and
// IEEE-754 maximum (not the C fmax / IEEE maxNum "ignore NaN" rule); on signed
// zeros it returns +0 for max(-0,+0), also matching numpy. The earlier
// `if v > m` form silently *ignored* NaNs (and a leading NaN poisoned the scan),
// diverging from numpy; this is the corrected, documented semantics.
//
// It uses the BUILTIN max, not math.Max: both have identical NaN/signed-zero
// semantics, but the builtin lowers to the hardware FMAXD intrinsic on arm64
// (and the equivalent elsewhere) — math.Max is an un-intrinsified call that is
// ~12x slower in this hot loop. The SSE2 maxSIMD kernel (MAXPD + NaN scan) is
// validated bit-identical to this oracle.
func Max(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		m = max(m, v)
	}
	return m
}

// Min returns the minimum element of a, which must be non-empty. Like Max it is
// NaN-propagating (any NaN -> NaN) and uses the builtin min (FMIND intrinsic),
// matching numpy.min including min(-0,+0) = -0.
func Min(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		m = min(m, v)
	}
	return m
}

// maxUnrolled is the fast NaN-propagating max reduction used by the non-amd64
// SIMD dispatch (arm64 and the four scalar arches). It keeps FOUR independent
// builtin-max accumulators so the dependency chain is broken — the builtin max
// lowers to the hardware FMAXD on arm64, and four parallel chains hide its
// latency (~3.6x over a single accumulator). max is associative for the
// NaN-propagating rule, so the result is bit-identical to the serial Max oracle
// (any NaN in any lane -> NaN; same extreme otherwise). a is non-empty.
func maxUnrolled(a []float64) float64 {
	n := len(a)
	if n < 4 {
		return Max(a)
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	i := 4
	for ; i+4 <= n; i += 4 {
		m0 = max(m0, a[i])
		m1 = max(m1, a[i+1])
		m2 = max(m2, a[i+2])
		m3 = max(m3, a[i+3])
	}
	m := max(max(m0, m1), max(m2, m3))
	for ; i < n; i++ {
		m = max(m, a[i])
	}
	return m
}

// minUnrolled is the dual of maxUnrolled (four independent builtin-min
// accumulators, FMIND on arm64); bit-identical to the serial Min oracle.
func minUnrolled(a []float64) float64 {
	n := len(a)
	if n < 4 {
		return Min(a)
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	i := 4
	for ; i+4 <= n; i += 4 {
		m0 = min(m0, a[i])
		m1 = min(m1, a[i+1])
		m2 = min(m2, a[i+2])
		m3 = min(m3, a[i+3])
	}
	m := min(min(m0, m1), min(m2, m3))
	for ; i < n; i++ {
		m = min(m, a[i])
	}
	return m
}

// Abs returns the absolute value of x. It exists so the parent package can
// route Abs through Map without importing math directly.
func Abs(x float64) float64 { return math.Abs(x) }

// blockMaxN is the column count up to which MatMul uses the 4-row register-
// blocked micro-kernel. Below it, holding four destination rows in registers and
// reusing each loaded b value four times is a ~2x win over the plain ikj kernel.
// At and above it the four destination rows (n*8 bytes apart) start colliding in
// the L1 cache on power-of-two widths, which erases the gain, so MatMul switches
// to the autovectorized ikj kernel — whose single contiguous AXPY inner loop the
// Go compiler vectorizes cleanly and which is memory-efficient at large n. The
// threshold is measured on the arm64 bench VM (the win holds through n=224, the
// collision sets in by n=256); it is a var so tests can pin both paths.
var blockMaxN = 224

// MatMul computes the (m x n) row-major matrix product dst = a (m x k) times
// b (k x n), where all three are flat contiguous []float64. dst must be zeroed
// by the caller. It dispatches between a register-blocked micro-kernel (small n)
// and the autovectorized ikj kernel (large n); both produce the identical result
// in the identical ikj summation order, so the choice is purely a speed/cache
// trade-off and never changes the values.
func MatMul(dst, a, b []float64, m, k, n int) {
	if n <= blockMaxN {
		matMulBlocked(dst, a, b, m, k, n)
		return
	}
	matMulIKJ(dst, a, b, m, k, n)
}

// matMulIKJ is the reference ikj-order GEMM: for each output row i and each k
// index p it does the contiguous AXPY dst[i][:] += a[i][p] * b[p][:]. The inner
// loop is unit-stride in both b and dst, the shape the Go compiler auto-
// vectorizes and a SIMD FMA kernel would want. It is the large-n path and the
// correctness reference the blocked kernel is validated against.
func matMulIKJ(dst, a, b []float64, m, k, n int) {
	for i := 0; i < m; i++ {
		dstRow := dst[i*n : i*n+n]
		for p := 0; p < k; p++ {
			av := a[i*k+p]
			bRow := b[p*n : p*n+n]
			for j := 0; j < n; j++ {
				dstRow[j] += av * bRow[j]
			}
		}
	}
}

// matMulBlocked is the 4-row register-blocked GEMM. It accumulates FOUR output
// rows at once: each b[p][j] is loaded once and fused into all four rows, so the
// b traffic and loop overhead are quartered versus the ikj kernel. The arithmetic
// is the same ikj order per row (dst[i][:] += sum_p a[i][p]*b[p][:]), so the
// result is bit-identical to matMulIKJ. Rows past the last multiple of four fall
// back to the single-row ikj body. Used only for n <= blockMaxN, where the four
// destination rows stay cache-resident.
func matMulBlocked(dst, a, b []float64, m, k, n int) {
	i := 0
	for ; i+4 <= m; i += 4 {
		d0 := dst[i*n : i*n+n]
		d1 := dst[(i+1)*n : (i+1)*n+n]
		d2 := dst[(i+2)*n : (i+2)*n+n]
		d3 := dst[(i+3)*n : (i+3)*n+n]
		for p := 0; p < k; p++ {
			a0 := a[i*k+p]
			a1 := a[(i+1)*k+p]
			a2 := a[(i+2)*k+p]
			a3 := a[(i+3)*k+p]
			bRow := b[p*n : p*n+n]
			for j, bv := range bRow {
				d0[j] += a0 * bv
				d1[j] += a1 * bv
				d2[j] += a2 * bv
				d3[j] += a3 * bv
			}
		}
	}
	for ; i < m; i++ {
		dstRow := dst[i*n : i*n+n]
		for p := 0; p < k; p++ {
			av := a[i*k+p]
			bRow := b[p*n : p*n+n]
			for j := 0; j < n; j++ {
				dstRow[j] += av * bRow[j]
			}
		}
	}
}

// Dot1D returns the inner product sum(a[i]*b[i]) of two equal-length vectors.
func Dot1D(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Axis reductions.
//
// The parent package materialises a strided view into a contiguous []float64
// laid out conceptually as [outer][axisLen][inner], i.e. the reduced axis sits
// in the middle and `inner` trailing elements are contiguous. Each kernel
// reduces the middle axis, writing outer*inner results into dst (also laid out
// as [outer][inner]).
//
// The innermost loop over `inner` walks contiguous memory with unit stride,
// which is the shape a SIMD kernel wants: Phase 1 can replace each step (the
// `inner`-length combine of dst[base:] with src[off:]) with a vectorised
// fused operation across all six 64-bit targets, leaving this scalar version as
// the reference and fallback. axisLen >= 1 is guaranteed by the caller.

// SumAxis reduces the middle axis by summation.
func SumAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			dst[base+i] = src[block+i]
		}
		for k := 1; k < axisLen; k++ {
			off := block + k*inner
			for i := 0; i < inner; i++ {
				dst[base+i] += src[off+i]
			}
		}
	}
}

// ProdAxis reduces the middle axis by multiplication.
func ProdAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			dst[base+i] = src[block+i]
		}
		for k := 1; k < axisLen; k++ {
			off := block + k*inner
			for i := 0; i < inner; i++ {
				dst[base+i] *= src[off+i]
			}
		}
	}
}

// MaxAxis reduces the middle axis by taking the maximum.
func MaxAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			dst[base+i] = src[block+i]
		}
		for k := 1; k < axisLen; k++ {
			off := block + k*inner
			for i := 0; i < inner; i++ {
				if v := src[off+i]; v > dst[base+i] {
					dst[base+i] = v
				}
			}
		}
	}
}

// MinAxis reduces the middle axis by taking the minimum.
func MinAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			dst[base+i] = src[block+i]
		}
		for k := 1; k < axisLen; k++ {
			off := block + k*inner
			for i := 0; i < inner; i++ {
				if v := src[off+i]; v < dst[base+i] {
					dst[base+i] = v
				}
			}
		}
	}
}

// ArgMax returns the index of the first maximum element of a (non-empty),
// matching numpy.argmax: ties go to the lowest index.
func ArgMax(a []float64) int {
	best, bi := a[0], 0
	for i, v := range a[1:] {
		if v > best {
			best, bi = v, i+1
		}
	}
	return bi
}

// ArgMin returns the index of the first minimum element of a (non-empty),
// matching numpy.argmin: ties go to the lowest index.
func ArgMin(a []float64) int {
	best, bi := a[0], 0
	for i, v := range a[1:] {
		if v < best {
			best, bi = v, i+1
		}
	}
	return bi
}

// ArgMaxAxis writes into dst the index (along the middle axis) of the first
// maximum for each [outer][inner] position. Layout matches the *Axis kernels.
func ArgMaxAxis(dst []float64, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			best := src[block+i]
			bi := 0
			for k := 1; k < axisLen; k++ {
				if v := src[block+k*inner+i]; v > best {
					best, bi = v, k
				}
			}
			dst[base+i] = float64(bi)
		}
	}
}

// ArgMinAxis writes into dst the index (along the middle axis) of the first
// minimum for each [outer][inner] position.
func ArgMinAxis(dst []float64, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			best := src[block+i]
			bi := 0
			for k := 1; k < axisLen; k++ {
				if v := src[block+k*inner+i]; v < best {
					best, bi = v, k
				}
			}
			dst[base+i] = float64(bi)
		}
	}
}

// CumSumAxis writes the cumulative sum along the middle axis into dst (same
// shape as src), matching numpy.cumsum along an axis. Layout is
// [outer][axisLen][inner] as for the reductions.
func CumSumAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			acc := 0.0
			for k := 0; k < axisLen; k++ {
				p := block + k*inner + i
				acc += src[p]
				dst[p] = acc
			}
		}
	}
}

// CumProdAxis writes the cumulative product along the middle axis into dst.
func CumProdAxis(dst, src []float64, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		block := o * axisLen * inner
		for i := 0; i < inner; i++ {
			acc := 1.0
			for k := 0; k < axisLen; k++ {
				p := block + k*inner + i
				acc *= src[p]
				dst[p] = acc
			}
		}
	}
}

// Clip writes into dst each src element limited to [lo, hi], matching
// numpy.clip. A NaN bound or element follows Go's min/max comparison order;
// callers pass lo <= hi.
func Clip(dst, src []float64, lo, hi float64) {
	for i, v := range src {
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = v
	}
}

// Where writes into dst the value t[i] where cond[i] is non-zero (truthy),
// else f[i] — the elementwise numpy.where over already-broadcast operands.
func Where(dst, cond, t, f []float64) {
	for i := range dst {
		if cond[i] != 0 {
			dst[i] = t[i]
		} else {
			dst[i] = f[i]
		}
	}
}
