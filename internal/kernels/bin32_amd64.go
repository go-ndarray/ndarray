package kernels

// bin32 runs the SSE2 kernel for op (sum_amd64.s: ADDPS, SUBPS, MULPS,
// DIVPS, 16 floats per iteration). SSE2 is the amd64 baseline. dst, a and b
// have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	run32([...]bin32Func{add32SSE2, sub32SSE2, mul32SSE2, div32SSE2}[op], dst, a, b)
}

// sqrt32 runs SQRTPS (sum_amd64.s). dst and src have the same length, at
// least 1.
func sqrt32(dst, src []float32) { sqrt32SSE2(&dst[0], &src[0], len(dst)) }

//go:noescape
func sqrt32SSE2(dst, src *float32, n int)

//go:noescape
func add32SSE2(dst, a, b *float32, n int)

//go:noescape
func sub32SSE2(dst, a, b *float32, n int)

//go:noescape
func mul32SSE2(dst, a, b *float32, n int)

//go:noescape
func div32SSE2(dst, a, b *float32, n int)
