package kernels

// addBlocks8F32 runs the NEON kernel (sum_arm64.s): the eight accumulators in
// two .S4 registers, one VFADD per register per block. len(a) is a positive
// multiple of 8 (AddBlocks8F32 checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8NEON(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8NEON(r, a *float32, blocks int)
