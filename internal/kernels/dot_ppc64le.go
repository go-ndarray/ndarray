package kernels

// dotSIMD is the inner product of two equal-length slices (VSX kernel, four
// fused multiply-add accumulators).
func dotSIMD(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1] // the kernel reads len(a) of b
	return dotVSX(&a[0], &b[0], len(a))
}

//go:noescape
func dotVSX(a, b *float64, n int) float64
