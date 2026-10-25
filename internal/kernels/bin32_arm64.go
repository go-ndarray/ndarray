package kernels

// bin32 runs the NEON kernel for op (sum_arm64.s: VFADD, VFSUB, VFMUL, VFDIV
// on .S4, 16 floats per iteration). NEON is the arm64 baseline. dst, a and b
// have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	run32([...]bin32Func{add32NEON, sub32NEON, mul32NEON, div32NEON}[op], dst, a, b)
}

// sqrt32 runs VFSQRT on .S4 (sum_arm64.s). dst and src have the same length,
// at least 1.
func sqrt32(dst, src []float32) { sqrt32NEON(&dst[0], &src[0], len(dst)) }

//go:noescape
func sqrt32NEON(dst, src *float32, n int)

//go:noescape
func add32NEON(dst, a, b *float32, n int)

//go:noescape
func sub32NEON(dst, a, b *float32, n int)

//go:noescape
func mul32NEON(dst, a, b *float32, n int)

//go:noescape
func div32NEON(dst, a, b *float32, n int)
