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
func Max(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// Min returns the minimum element of a, which must be non-empty.
func Min(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

// Abs returns the absolute value of x. It exists so the parent package can
// route Abs through Map without importing math directly.
func Abs(x float64) float64 { return math.Abs(x) }

// MatMul computes the (m x n) row-major matrix product dst = a (m x k) times
// b (k x n), where all three are flat contiguous []float64. dst must be
// zeroed by the caller. The innermost loop over n is unit-stride contiguous in
// both b and dst — the shape a SIMD FMA kernel wants — so Phase 1 can replace
// the inner accumulation across all six 64-bit targets, leaving this ikj-order
// scalar version as the reference and fallback.
func MatMul(dst, a, b []float64, m, k, n int) {
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
