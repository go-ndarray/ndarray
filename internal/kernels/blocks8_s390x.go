package kernels

// addBlocks8F32 runs the VXE kernel (sum_s390x.s), the eight accumulators in
// two vector registers, one VFASB per register per block, on a z14 or later
// (VFASB is vector-enhancements facility 1, the useVXE gate of max/min); the
// scalar loop otherwise. len(a) is a positive multiple of 8 (AddBlocks8F32
// checks).
func addBlocks8F32(r *[8]float32, a []float32) {
	if !useVXE {
		addBlocks8F32Go(r, a)
		return
	}
	_ = a[len(a)-1] // the kernel reads len(a) of a
	blocks8VXE(&r[0], &a[0], len(a)/8)
}

//go:noescape
func blocks8VXE(r, a *float32, blocks int)
