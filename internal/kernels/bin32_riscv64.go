package kernels

// bin32 runs the RVV kernel for op (sum_riscv64.s: vfadd.vv, vfsub.vv,
// vfmul.vv, vfdiv.vv on 32-bit elements, strip-mined) when the CPU reports V;
// the scalar loop otherwise. dst, a and b have the same length, at least 1.
func bin32(op int, dst, a, b []float32) {
	if !useRVV {
		bin32Go(op, dst, a, b)
		return
	}
	run32([...]bin32Func{add32RVV, sub32RVV, mul32RVV, div32RVV}[op], dst, a, b)
}

// sqrt32 runs vfsqrt.v (sum_riscv64.s) when the CPU reports V; the scalar
// loop otherwise. dst and src have the same length, at least 1.
func sqrt32(dst, src []float32) {
	if !useRVV {
		sqrt32Go(dst, src)
		return
	}
	sqrt32RVV(&dst[0], &src[0], len(dst))
}

//go:noescape
func sqrt32RVV(dst, src *float32, n int)

//go:noescape
func add32RVV(dst, a, b *float32, n int)

//go:noescape
func sub32RVV(dst, a, b *float32, n int)

//go:noescape
func mul32RVV(dst, a, b *float32, n int)

//go:noescape
func div32RVV(dst, a, b *float32, n int)
