package kernels

import (
	"math"

	"github.com/go-simd/floats"
)

// Exp, Log and their float32 forms are go-simd/floats': K1 of the go-scikit
// performance campaign moved the kernels there (they were written here:
// go-ndarray#94 and #96 for AVX2, #95 for NEON), so that a kernel lives once.
// What stays here is the parallel split above ParThreshold and Log10.
//
//   - floats.Exp / floats.Log: float64, 0.511 / 0.519 ULP (Arm optimized-routines);
//   - floats.Float32Exp / floats.Float32Log: correctly rounded float32, proven
//     over every float32 input in go-simd/floats (TestExp32CorrectlyRounded,
//     TestLog32CorrectlyRounded), on AVX2 (amd64) and NEON (arm64) kernels.

// Exp writes exp(src[i]) into dst[i].
func Exp(dst, src []float64) { floats.Exp(dst, src) }

// Log writes log(src[i]) into dst[i].
func Log(dst, src []float64) { floats.Log(dst, src) }

// Exp32 writes the correctly rounded exp(src[i]) into dst[i].
func Exp32(dst, src []float32) { floats.Float32Exp(dst, src) }

// Log32 writes the correctly rounded log(src[i]) into dst[i].
func Log32(dst, src []float32) { floats.Float32Log(dst, src) }

// Log10 writes log10(src[i]) into dst[i] as log(x)*(1/ln10), the formula of
// math.Log10, on floats.Log.
func Log10(dst, src []float64) {
	Log(dst, src)
	for i := range dst {
		dst[i] *= 1 / math.Ln10
	}
}

// ExpP is Exp, split across cores above ParThreshold.
func ExpP(dst, src []float64) {
	n := len(dst)
	if n < ParThreshold {
		Exp(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { Exp(dst[lo:hi], src[lo:hi]) })
}

// LogP is Log, split across cores above ParThreshold.
func LogP(dst, src []float64) {
	n := len(dst)
	if n < ParThreshold {
		Log(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { Log(dst[lo:hi], src[lo:hi]) })
}

// Log10P is Log10, split across cores above ParThreshold.
func Log10P(dst, src []float64) {
	n := len(dst)
	if n < ParThreshold {
		Log10(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { Log10(dst[lo:hi], src[lo:hi]) })
}

// Exp32P is Exp32, split across cores above ParThreshold.
func Exp32P(dst, src []float32) { split32(Exp32, dst, src) }

// Log32P is Log32, split across cores above ParThreshold.
func Log32P(dst, src []float32) { split32(Log32, dst, src) }

func split32(f func(dst, src []float32), dst, src []float32) {
	n := len(dst)
	if n < ParThreshold {
		f(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { f(dst[lo:hi], src[lo:hi]) })
}
