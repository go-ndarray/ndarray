//go:build ppc64le || loong64 || riscv64 || s390x

package kernels

import "github.com/go-simd/floats"

// On ppc64le (VSX), loong64 (LASX), riscv64 (RVV) and s390x (VX) the float64
// elementwise and square-root kernels are go-simd/floats' since v0.10.0
// (floats#30): they moved there from this package's sum_<arch>.s, with the same
// instruction sequences, and floats gates them on the CPU feature itself. Each is
// bit-identical to the scalar oracle (Add, Sub, Mul, Div, sqrtScalar).

// sqrtSIMD writes sqrt(src[i]) into dst[i] (correctly rounded, like math.Sqrt).
func sqrtSIMD(dst, src []float64) { floats.Sqrt(dst, src[:len(dst)]) }

// addBin/subBin/mulBin/divBin write a[i] OP b[i] into dst[i]; len(dst) elements.
func addBin(dst, a, b []float64) { floats.Add(dst, a[:len(dst)], b[:len(dst)]) }
func subBin(dst, a, b []float64) { floats.Sub(dst, a[:len(dst)], b[:len(dst)]) }
func mulBin(dst, a, b []float64) { floats.Mul(dst, a[:len(dst)], b[:len(dst)]) }
func divBin(dst, a, b []float64) { floats.Div(dst, a[:len(dst)], b[:len(dst)]) }
