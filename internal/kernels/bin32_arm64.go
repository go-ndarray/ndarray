package kernels

// bin32 runs the NEON kernel for op (sum_arm64.s: VFADD, VFSUB, VFMUL, VFDIV
// on .S4, 16 floats per iteration). NEON is the arm64 baseline. dst, a and b
// have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	run32([...]bin32Func{add32NEON, sub32NEON, mul32NEON, div32NEON}[op], dst, a, b)
}

//go:noescape
func add32NEON(dst, a, b *float32, n int)

//go:noescape
func sub32NEON(dst, a, b *float32, n int)

//go:noescape
func mul32NEON(dst, a, b *float32, n int)

//go:noescape
func div32NEON(dst, a, b *float32, n int)
