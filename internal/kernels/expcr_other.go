//go:build !amd64 && !arm64

package kernels

// exp32Fast is the correctly rounded float32 exp: expCR in Go on these
// architectures (no SIMD kernel yet).
func exp32Fast(dst, src []float32) { exp32CR(dst, src) }

// log32Fast is the correctly rounded float32 log: logCR in Go here.
func log32Fast(dst, src []float32) { logCRSlice(dst, src) }
