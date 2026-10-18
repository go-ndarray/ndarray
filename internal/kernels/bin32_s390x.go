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

//go:noescape
func add32VXE(dst, a, b *float32, n int)

//go:noescape
func sub32VXE(dst, a, b *float32, n int)

//go:noescape
func mul32VXE(dst, a, b *float32, n int)

//go:noescape
func div32VXE(dst, a, b *float32, n int)
