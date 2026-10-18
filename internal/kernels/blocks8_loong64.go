package kernels

// addBlocks8F32 runs the LASX kernel (sum_loong64.s), the eight accumulators
// in one 256-bit register, when the CPU has LASX; the scalar loop otherwise.
// len(a) is a positive multiple of 8 (AddBlocks8F32 checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	if !useLASX {
		addBlocks8F32Go(r, a)
		return
	}
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8LASX(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8LASX(r, a *float32, blocks int)
