package kernels

// dotSIMD is the inner product of two equal-length slices (NEON kernel).
func dotSIMD(a, b []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	_ = b[len(a)-1] // the kernel reads len(a) of b
	return dotNEON(&a[0], &b[0], len(a))
}

//go:noescape
func dotNEON(a, b *float64, n int) float64
