package kernels

// addBlocks8F32 runs the LSX kernel (sum_loong64.s), the eight accumulators
// in two 128-bit registers, when the CPU has LASX (which every LASX CPU's LSX
// comes with); the scalar loop otherwise. len(a) is a positive multiple of 8
// (AddBlocks8F32 checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	if !useLASX {
		addBlocks8F32Go(r, a)
		return
	}
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8LSX(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8LSX(r, a *float32, blocks int)
