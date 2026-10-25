package kernels

// bin32 runs the VSX kernel for op (sum_ppc64le.s: xvaddsp, xvsubsp,
// xvmulsp, xvdivsp, 16 floats per iteration). VSX is the ppc64le baseline.
// dst, a and b have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	run32([...]bin32Func{add32VSX, sub32VSX, mul32VSX, div32VSX}[op], dst, a, b)
}

// sqrt32 runs xvsqrtsp (sum_ppc64le.s). dst and src have the same length, at
// least 1.
func sqrt32(dst, src []float32) { sqrt32VSX(&dst[0], &src[0], len(dst)) }

//go:noescape
func sqrt32VSX(dst, src *float32, n int)

//go:noescape
func add32VSX(dst, a, b *float32, n int)

//go:noescape
func sub32VSX(dst, a, b *float32, n int)

//go:noescape
func mul32VSX(dst, a, b *float32, n int)

//go:noescape
func div32VSX(dst, a, b *float32, n int)
