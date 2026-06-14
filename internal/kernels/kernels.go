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
