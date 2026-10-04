// Package kernels holds the inner numeric loops for the ndarray package.
//
// Every loop here is a portable, pure-Go (CGO=0) scalar kernel. They are kept
// behind this small, contiguous-slice API so that SIMD variants drop in without
// changing callers or tests: go-asmgen-generated kernels replace the hot ones
// on amd64, arm64, ppc64le and loong64 (the *_<arch> files), while these
// scalar versions remain the reference and the fallback elsewhere.
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

// Maximum writes the pairwise maximum of a[i] and b[i] into dst[i]. It is
// NaN-propagating like numpy.maximum: it uses the builtin max, not math.Max,
// because math.Max returns +Inf for max(NaN, +Inf) where numpy returns NaN.
func Maximum(dst, a, b []float64) {
	for i := range dst {
		dst[i] = max(a[i], b[i])
	}
}

// Minimum writes the pairwise minimum of a[i] and b[i] into dst[i],
// NaN-propagating like numpy.minimum (builtin min; math.Min returns -Inf for
// min(NaN, -Inf)).
func Minimum(dst, a, b []float64) {
	for i := range dst {
		dst[i] = min(a[i], b[i])
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

// Abs returns the absolute value of x. It exists so the parent package can
// route Abs through Map without importing math directly.
func Abs(x float64) float64 { return math.Abs(x) }

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
// reduces the middle axis for the inner columns [lo, hi) of every outer slab,
// writing those columns of dst (laid out as [outer][inner]). A column band is
// contiguous within each row, so RunAxisP can hand disjoint bands to different
// workers when there are too few outer slabs to split (an axis-0 reduction has
// one). The full reduction is the band [0, inner). axisLen >= 1 is guaranteed
// by the caller.

// rowRun is the band width from which a row combine goes through the SIMD
// elementwise kernel; below it the call costs more than the loop.
const rowRun = 16

// SumAxis reduces the middle axis by summation. Each row of the band is added
// into dst with the elementwise SIMD add, element by element in axis order, so
// the result is the sequential sum. With inner == 1 every output is the sum of
// one contiguous row, taken with the lane-parallel SIMD sum (a regrouping, as
// numpy's pairwise sum is): a sequential sum there is latency-bound.
func SumAxis(dst, src []float64, outer, axisLen, inner, lo, hi int) {
	if inner == 1 {
		for o := 0; o < outer; o++ {
			dst[o] = sumSIMD(src[o*axisLen : o*axisLen+axisLen])
		}
		return
	}
	for o := 0; o < outer; o++ {
		d := dst[o*inner+lo : o*inner+hi]
		block := o * axisLen * inner
		copy(d, src[block+lo:block+hi])
		for k := 1; k < axisLen; k++ {
			r := src[block+k*inner+lo : block+k*inner+hi]
			if len(d) >= rowRun {
				addBin(d, d, r)
				continue
			}
			for i := range d {
				d[i] += r[i]
			}
		}
	}
}

// ProdAxis reduces the middle axis by multiplication, in axis order.
func ProdAxis(dst, src []float64, outer, axisLen, inner, lo, hi int) {
	for o := 0; o < outer; o++ {
		d := dst[o*inner+lo : o*inner+hi]
		block := o * axisLen * inner
		copy(d, src[block+lo:block+hi])
		for k := 1; k < axisLen; k++ {
			r := src[block+k*inner+lo : block+k*inner+hi]
			if len(d) >= rowRun {
				mulBin(d, d, r)
				continue
			}
			for i := range d {
				d[i] *= r[i]
			}
		}
	}
}

// MaxAxis reduces the middle axis by taking the maximum. Like Max it is
// NaN-propagating (numpy.max(axis=...)); max is exact and associative, so the
// inner == 1 row reduction through the SIMD max kernel is the same value.
func MaxAxis(dst, src []float64, outer, axisLen, inner, lo, hi int) {
	if inner == 1 {
		for o := 0; o < outer; o++ {
			dst[o] = maxSIMD(src[o*axisLen : o*axisLen+axisLen])
		}
		return
	}
	for o := 0; o < outer; o++ {
		d := dst[o*inner+lo : o*inner+hi]
		block := o * axisLen * inner
		copy(d, src[block+lo:block+hi])
		for k := 1; k < axisLen; k++ {
			r := src[block+k*inner+lo : block+k*inner+hi]
			for i := range d {
				d[i] = max(d[i], r[i])
			}
		}
	}
}

// MinAxis reduces the middle axis by taking the minimum, NaN-propagating
// (numpy.min(axis=...)).
func MinAxis(dst, src []float64, outer, axisLen, inner, lo, hi int) {
	if inner == 1 {
		for o := 0; o < outer; o++ {
			dst[o] = minSIMD(src[o*axisLen : o*axisLen+axisLen])
		}
		return
	}
	for o := 0; o < outer; o++ {
		d := dst[o*inner+lo : o*inner+hi]
		block := o * axisLen * inner
		copy(d, src[block+lo:block+hi])
		for k := 1; k < axisLen; k++ {
			r := src[block+k*inner+lo : block+k*inner+hi]
			for i := range d {
				d[i] = min(d[i], r[i])
			}
		}
	}
}

// ArgMax returns the index of the first maximum element of a (non-empty),
// matching numpy.argmax: ties go to the lowest index, and a NaN counts as the
// maximum, so the index of the first NaN wins (as Max returns NaN).
func ArgMax(a []float64) int {
	best, bi := a[0], 0
	if math.IsNaN(best) {
		return 0
	}
	for i, v := range a[1:] {
		if math.IsNaN(v) {
			return i + 1
		}
		if v > best {
			best, bi = v, i+1
		}
	}
	return bi
}

// ArgMin returns the index of the first minimum element of a (non-empty),
// matching numpy.argmin: ties go to the lowest index, and the first NaN wins.
func ArgMin(a []float64) int {
	best, bi := a[0], 0
	if math.IsNaN(best) {
		return 0
	}
	for i, v := range a[1:] {
		if math.IsNaN(v) {
			return i + 1
		}
		if v < best {
			best, bi = v, i+1
		}
	}
	return bi
}

// ArgMaxAxis writes into dst the index (along the middle axis) of the first
// maximum for each [outer][inner] position, or of the first NaN (see ArgMax).
// Layout matches the *Axis kernels.
func ArgMaxAxis(dst []float64, src []float64, outer, axisLen, inner, lo, hi int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := lo; i < hi; i++ {
			best := src[block+i]
			bi := 0
			for k := 1; k < axisLen && !math.IsNaN(best); k++ {
				if v := src[block+k*inner+i]; v > best || math.IsNaN(v) {
					best, bi = v, k
				}
			}
			dst[base+i] = float64(bi)
		}
	}
}

// ArgMinAxis writes into dst the index (along the middle axis) of the first
// minimum for each [outer][inner] position, or of the first NaN.
func ArgMinAxis(dst []float64, src []float64, outer, axisLen, inner, lo, hi int) {
	for o := 0; o < outer; o++ {
		base := o * inner
		block := o * axisLen * inner
		for i := lo; i < hi; i++ {
			best := src[block+i]
			bi := 0
			for k := 1; k < axisLen && !math.IsNaN(best); k++ {
				if v := src[block+k*inner+i]; v < best || math.IsNaN(v) {
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
