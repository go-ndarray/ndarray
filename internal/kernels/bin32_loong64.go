package kernels

// bin32 runs the LSX kernel for op (sum_loong64.s: vfadd.s, vfsub.s,
// vfmul.s, vfdiv.s over four 128-bit registers) when the CPU has LASX, which
// every LASX CPU's LSX comes with; the scalar loop otherwise. dst, a and b
// have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	if !useLASX {
		bin32Go(op, dst, a, b)
		return
	}
	run32([...]bin32Func{add32LSX, sub32LSX, mul32LSX, div32LSX}[op], dst, a, b)
}

//go:noescape
func add32LSX(dst, a, b *float32, n int)

//go:noescape
func sub32LSX(dst, a, b *float32, n int)

//go:noescape
func mul32LSX(dst, a, b *float32, n int)

//go:noescape
func div32LSX(dst, a, b *float32, n int)
