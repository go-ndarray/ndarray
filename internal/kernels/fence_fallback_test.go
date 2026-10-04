//go:build arm64 || ppc64le

package kernels

// forEachFallback has nothing to do: these kernels have no CPU-dependent
// alternative.
func forEachFallback(func()) {}
