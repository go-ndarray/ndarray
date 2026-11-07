//go:build !amd64

package kernels

// logFast is Log's Go loop on these architectures (no SIMD kernel yet).
func logFast(dst, src []float64) { logScalar(dst, src) }
