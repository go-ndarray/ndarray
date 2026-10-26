package kernels

// max32 runs the RVV kernel (sum_riscv64.s: vfmax.vv with a vmfne NaN count)
// when the CPU reports V, the four-chain Go loop otherwise.
func max32(a []float32) float32 {
	if !useRVV {
		return max32Go(a)
	}
	return max32RVV(&a[0], len(a))
}

// min32 is the dual of max32, with vfmin.vv.
func min32(a []float32) float32 {
	if !useRVV {
		return min32Go(a)
	}
	return min32RVV(&a[0], len(a))
}

//go:noescape
func max32RVV(a *float32, n int) float32

//go:noescape
func min32RVV(a *float32, n int) float32
