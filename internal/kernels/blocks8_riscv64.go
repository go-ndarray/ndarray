package kernels

// addBlocks8F32 runs the RVV kernel (sum_riscv64.s), the eight accumulators
// in a two-register group at vl = 8, when the CPU has the V extension; the
// scalar loop otherwise. len(a) is a positive multiple of 8 (AddBlocks8F32
// checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	if !useRVV {
		addBlocks8F32Go(r, a)
		return
	}
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8RVV(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8RVV(r, a *float32, blocks int)
