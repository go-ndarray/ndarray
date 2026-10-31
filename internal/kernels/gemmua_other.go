//go:build !amd64 && !arm64

package kernels

// gemmMicroUA is nil here: no unpacked-A float64 kernel, so A is packed.
var gemmMicroUA func(kc int, a []float64, lda int, b []float64, ldb int, dst []float64, ldc, mr int)
