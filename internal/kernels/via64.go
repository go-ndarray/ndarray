package kernels

// Correctly rounded float32 Exp and Log.
//
// float32 Exp and Log went through math.Exp and math.Log per element, 6-14x
// slower than NumPy's float32 SIMD on one Zen 3 core. Both now compute in
// float64 and round once, and both are proven correctly rounded over every
// float32 input (via64_exhaustive_test.go, expcr_test.go, logcr_test.go):
//
//   - Exp runs expCR (expcr.go), Arm's correctly rounded expf algorithm;
//   - Log runs logCR (logcr.go), CORE-MATH's correctly rounded logf;
//
// each on an AVX2 kernel where there is one (expcr_amd64.go, logcr_amd64.go),
// in Go elsewhere. Nothing is allocated.

const via64Chunk = 256

// Exp32 writes the correctly rounded exp(src[i]) into dst[i].
func Exp32(dst, src []float32) { exp32Fast(dst, src) }

// Log32 writes the correctly rounded log(src[i]) into dst[i].
func Log32(dst, src []float32) { log32Fast(dst, src) }

// Exp32P is Exp32, split across cores above ParThreshold.
func Exp32P(dst, src []float32) { split32(exp32Fast, dst, src) }

// Log32P is Log32, split across cores above ParThreshold.
func Log32P(dst, src []float32) { split32(log32Fast, dst, src) }

func split32(f func(dst, src []float32), dst, src []float32) {
	n := len(dst)
	if n < ParThreshold {
		f(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { f(dst[lo:hi], src[lo:hi]) })
}
