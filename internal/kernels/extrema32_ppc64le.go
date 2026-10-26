package kernels

import "math"

// max32 reduces whole blocks of 16 with xvmaxsp (sum_ppc64le.s), then folds
// the four lanes and the tail with Go's max. As for float64 (maxSIMD), NaN is
// reported separately by an integer magnitude test, and a zero result goes to
// the scalar loop for its sign.
func max32(a []float32) float32 {
	n := len(a) &^ 15
	if n == 0 {
		return max32Go(a)
	}
	var out [4]float32
	if max32VSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := max(max(out[0], out[1]), max(out[2], out[3]))
	for _, v := range a[n:] {
		m = max(m, v)
	}
	if m == 0 {
		return max32Go(a)
	}
	return m
}

// min32 is the dual of max32, with xvminsp.
func min32(a []float32) float32 {
	n := len(a) &^ 15
	if n == 0 {
		return min32Go(a)
	}
	var out [4]float32
	if min32VSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := min(min(out[0], out[1]), min(out[2], out[3]))
	for _, v := range a[n:] {
		m = min(m, v)
	}
	if m == 0 {
		return min32Go(a)
	}
	return m
}

//go:noescape
func max32VSX(a *float32, n int, out *[4]float32) int64

//go:noescape
func min32VSX(a *float32, n int, out *[4]float32) int64
