//go:build amd64 || arm64

package kernels

import "github.com/go-simd/floats"

// On amd64 (AVX2) and arm64 (NEON) the elementwise, square-root and extremum
// kernels are go-simd/floats': K1 of the go-scikit performance campaign moved
// them there, so that a kernel lives once, in the lowest library every user can
// import. Each is bit-identical to the scalar oracle (Add, Sub, Mul, Div,
// sqrtScalar, Max, Min of kernels.go) on every input, signed zeros and NaNs
// included; the tests of this package hold them to it.
//
// amd64 needs AVX2+FMA for the AVX2 kernels (floats' gate); on a CPU without it
// floats runs its SSE2 kernels (floats v0.10.0, floats#31), as this package's
// own SSE2 kernels did before v0.37.0.

// HaveReduceSIMD reports that this build ships SIMD reduction/elementwise
// kernels (so the dispatch wrappers route through them rather than the scalar
// oracle). True on amd64/arm64.
const HaveReduceSIMD = true

// sqrtSIMD writes sqrt(src[i]) into dst[i] (correctly rounded, like math.Sqrt).
func sqrtSIMD(dst, src []float64) { floats.Sqrt(dst, src[:len(dst)]) }

// maxSIMD returns the NaN-propagating maximum of a (non-empty), the serial Max
// oracle's value (max(-0, +0) = +0, as NumPy). floats.Max has that contract since
// v0.10.0 (floats#32), so no zero result needs resolving here any more.
func maxSIMD(a []float64) float64 { return floats.Max(a) }

// minSIMD returns the NaN-propagating minimum of a (non-empty) (min(-0, +0) =
// -0), floats.Min's contract since v0.10.0.
func minSIMD(a []float64) float64 { return floats.Min(a) }

// addBin/subBin/mulBin/divBin write a[i] OP b[i] into dst[i]; len(dst) elements.
func addBin(dst, a, b []float64) { floats.Add(dst, a[:len(dst)], b[:len(dst)]) }
func subBin(dst, a, b []float64) { floats.Sub(dst, a[:len(dst)], b[:len(dst)]) }
func mulBin(dst, a, b []float64) { floats.Mul(dst, a[:len(dst)], b[:len(dst)]) }
func divBin(dst, a, b []float64) { floats.Div(dst, a[:len(dst)], b[:len(dst)]) }
