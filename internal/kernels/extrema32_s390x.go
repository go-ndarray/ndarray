package kernels

// max32 runs the VXE kernel (sum_s390x.s: VFMAXSB in Java mode, NaN-propagating
// with -0 below +0) from 16 elements up on a z14 or later, the four-chain Go
// loop otherwise.
func max32(a []float32) float32 {
	if !useVXE || len(a) < 16 {
		return max32Go(a)
	}
	return max32VXE(&a[0], len(a))
}

// min32 is the dual of max32, with VFMINSB.
func min32(a []float32) float32 {
	if !useVXE || len(a) < 16 {
		return min32Go(a)
	}
	return min32VXE(&a[0], len(a))
}

//go:noescape
func max32VXE(a *float32, n int) float32

//go:noescape
func min32VXE(a *float32, n int) float32
