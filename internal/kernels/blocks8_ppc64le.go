package kernels

// addBlocks8F32 runs the VSX kernel (sum_ppc64le.s): the eight accumulators in
// two VSX registers, one XVADDSP per register per block. VSX is the ppc64le
// baseline. len(a) is a positive multiple of 8 (AddBlocks8F32 checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8VSX(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8VSX(r, a *float32, blocks int)
