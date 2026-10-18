package kernels

// AddBlocks8F32 adds a, a whole number of blocks of eight float32, into the
// eight accumulators r lane by lane: for each block i in order,
// r[j] += a[8i+j]. It is the inner loop of NumPy's pairwise sum
// (FLOAT_pairwise_sum, whose eight accumulators exist "to allow vectorization
// without changing summation ordering"), so a vector kernel gives exactly the
// same sums as the scalar loop: one IEEE float32 addition per lane per block,
// in the same order. It panics if len(a) is not a multiple of 8.
func AddBlocks8F32(r *[8]float32, a []float32) {
	if len(a)%8 != 0 {
		panic("kernels: AddBlocks8F32 needs whole blocks of 8")
	}
	if len(a) == 0 {
		return
	}
	addBlocks8F32(r, a)
}

// addBlocks8F32Go is the scalar oracle the kernels are tested against, and
// the implementation on the architectures without one.
func addBlocks8F32Go(r *[8]float32, a []float32) {
	for i := 0; i+8 <= len(a); i += 8 {
		b := a[i : i+8 : i+8]
		r[0] += b[0]
		r[1] += b[1]
		r[2] += b[2]
		r[3] += b[3]
		r[4] += b[4]
		r[5] += b[5]
		r[6] += b[6]
		r[7] += b[7]
	}
}
