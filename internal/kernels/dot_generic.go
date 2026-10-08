//go:build !arm64 && !amd64 && !ppc64le && !loong64 && !riscv64 && !s390x

package kernels

// dotSIMD is dotRange here: no vector-double kernel on these targets.
func dotSIMD(a, b []float64) float64 { return dotRange(a, b) }
