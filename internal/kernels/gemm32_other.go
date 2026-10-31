//go:build !amd64 && !arm64 && !ppc64le && !s390x && !riscv64

package kernels

// gemm32 is nil here: no float32 micro-kernel yet, so HasGemm32 is false and
// float32 products are computed in float64.
var gemm32 *gemmKernel[float32]
