//go:build !amd64

package kernels

// exp32Fast is the correctly rounded float32 exp: expCR in Go on these
// architectures (no SIMD kernel yet).
func exp32Fast(dst, src []float32) { exp32CR(dst, src) }
