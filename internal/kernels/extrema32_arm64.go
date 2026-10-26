package kernels

// max32 runs the NEON kernel (sum_arm64.s: VFMAX on .S4, four accumulators)
// from 16 elements up, the four-chain Go loop below that. VFMAX propagates NaN
// and orders -0 below +0, so no correction is needed.
func max32(a []float32) float32 {
	if len(a) < 16 {
		return max32Go(a)
	}
	return max32NEON(&a[0], len(a))
}

// min32 is the dual of max32, with VFMIN.
func min32(a []float32) float32 {
	if len(a) < 16 {
		return min32Go(a)
	}
	return min32NEON(&a[0], len(a))
}

//go:noescape
func max32NEON(a *float32, n int) float32

//go:noescape
func min32NEON(a *float32, n int) float32
