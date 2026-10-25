package kernels

// bin32 runs the vector kernel for op (sum_s390x.s: vfasb, vfssb, vfmsb,
// vfdsb, 16 floats per iteration) on a z14 or later, whose vector-enhancements
// facility 1 brings them (the useVXE gate of max/min); the scalar loop
// otherwise. dst, a and b have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	if !useVXE {
		bin32Go(op, dst, a, b)
		return
	}
	run32([...]bin32Func{add32VXE, sub32VXE, mul32VXE, div32VXE}[op], dst, a, b)
}

// sqrt32 runs vfsqsb (sum_s390x.s) on a z14 or later; the scalar loop
// otherwise. dst and src have the same length, at least 1.
func sqrt32(dst, src []float32) {
	if !useVXE {
		sqrt32Go(dst, src)
		return
	}
	sqrt32VXE(&dst[0], &src[0], len(dst))
}

//go:noescape
func sqrt32VXE(dst, src *float32, n int)

//go:noescape
func add32VXE(dst, a, b *float32, n int)

//go:noescape
func sub32VXE(dst, a, b *float32, n int)

//go:noescape
func mul32VXE(dst, a, b *float32, n int)

//go:noescape
func div32VXE(dst, a, b *float32, n int)
