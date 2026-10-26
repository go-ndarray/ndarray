//go:build !amd64 && !arm64 && !ppc64le && !riscv64 && !s390x

package kernels

// The float32 extrema run the four-chain Go loops on the architectures
// without a vector kernel for them yet.

func max32(a []float32) float32 { return max32Go(a) }
func min32(a []float32) float32 { return min32Go(a) }
