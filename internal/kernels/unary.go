package kernels

import "github.com/go-simd/floats"

// Unary float64 kernels from go-simd/floats (v0.23.0), parallelised above
// ParThreshold like the other elementwise kernels. Each is bit-identical to the
// scalar expression it replaces (math.Abs, -x, x*x, math.Floor, math.Ceil), a
// signalling NaN possibly coming back quieted.

// unaryP runs k over dst/src, serially below ParThreshold and fanned across
// cores above it. len(src) >= len(dst) is required, as for the kernels.
func unaryP(k func(dst, src []float64), dst, src []float64) {
	n := len(dst)
	if n < ParThreshold {
		k(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) {
		k(dst[lo:hi], src[lo:hi])
	})
}

// AbsP writes |src[i]| into dst[i].
func AbsP(dst, src []float64) { unaryP(floats.Abs, dst, src) }

// NegP writes -src[i] into dst[i].
func NegP(dst, src []float64) { unaryP(floats.Neg, dst, src) }

// SquareP writes src[i]*src[i] into dst[i].
func SquareP(dst, src []float64) { unaryP(floats.Square, dst, src) }

// FloorP writes math.Floor(src[i]) into dst[i].
func FloorP(dst, src []float64) { unaryP(floats.Floor, dst, src) }

// RoundEvenP writes math.RoundToEven(src[i]) into dst[i].
func RoundEvenP(dst, src []float64) { unaryP(floats.RoundEven, dst, src) }

// CeilP writes math.Ceil(src[i]) into dst[i].
func CeilP(dst, src []float64) { unaryP(floats.Ceil, dst, src) }
