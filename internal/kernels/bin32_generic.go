//go:build !amd64 && !arm64 && !loong64 && !ppc64le && !riscv64 && !s390x

package kernels

// bin32 is the scalar loop on the architectures without float32 elementwise
// kernels. dst, a and b have the same length, at least 1.
func bin32(op int, dst, a, b []float32) { bin32Go(op, dst, a, b) }

// sqrt32 is the scalar loop there too.
func sqrt32(dst, src []float32) { sqrt32Go(dst, src) }
