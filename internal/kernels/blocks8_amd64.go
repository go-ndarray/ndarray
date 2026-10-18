package kernels

// addBlocks8F32 runs the SSE2 kernel (sum_amd64.s): the eight accumulators in
// two XMM registers, one ADDPS per register per block. len(a) is a positive
// multiple of 8 (AddBlocks8F32 checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8SSE2(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8SSE2(r, a *float32, blocks int)
