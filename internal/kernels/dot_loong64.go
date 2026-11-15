package kernels

// dotSIMD is the inner product (LASX kernel, four fused accumulators).
func dotSIMD(a, b []float64) float64 {
	if !useLASX || len(a) == 0 {
		return dotRange(a, b)
	}
	_ = b[len(a)-1] // the kernel reads len(a) of b
	return dotLASX(&a[0], &b[0], len(a))
}

//go:noescape
func dotLASX(a, b *float64, n int) float64
