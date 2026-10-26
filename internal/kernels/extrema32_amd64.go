package kernels

// max32 runs the SSE2 kernel (sum_amd64.s: MAXPS, four accumulators, a
// CMPPS-unordered NaN mask). MAXPS returns its source operand on a zero tie,
// so a zero result may have the wrong sign: that case, which needs every
// element <= 0 with a zero among them, goes to the scalar loop (Go's max: +0
// if any +0), as maxSIMD does for float64.
func max32(a []float32) float32 {
	if r := max32SSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return max32Go(a)
}

// min32 is the dual of max32 (a zero result: -0 if any -0).
func min32(a []float32) float32 {
	if r := min32SSE2(&a[0], len(a)); r != 0 {
		return r
	}
	return min32Go(a)
}

//go:noescape
func max32SSE2(a *float32, n int) float32

//go:noescape
func min32SSE2(a *float32, n int) float32
