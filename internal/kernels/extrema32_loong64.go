package kernels

import "math"

// max32 reduces whole blocks of 16 with the LSX kernel on the floats' bits in
// integer order (see fromOrdered32), then folds the four lanes and the tail
// with Go's max. A NaN anywhere, reported by the kernel, is NaN.
func max32(a []float32) float32 {
	n := len(a) &^ 15
	if !useLASX || n == 0 {
		return max32Go(a)
	}
	var out [4]int32
	if max32LSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := max(max(fromOrdered32(out[0]), fromOrdered32(out[1])), max(fromOrdered32(out[2]), fromOrdered32(out[3])))
	for _, v := range a[n:] {
		m = max(m, v)
	}
	return m
}

// min32 is the dual of max32.
func min32(a []float32) float32 {
	n := len(a) &^ 15
	if !useLASX || n == 0 {
		return min32Go(a)
	}
	var out [4]int32
	if min32LSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := min(min(fromOrdered32(out[0]), fromOrdered32(out[1])), min(fromOrdered32(out[2]), fromOrdered32(out[3])))
	for _, v := range a[n:] {
		m = min(m, v)
	}
	return m
}

// fromOrdered32 undoes the kernels' order map t = x XOR ((x >>a 31) >>l 1).
func fromOrdered32(t int32) float32 {
	return math.Float32frombits(uint32(t ^ int32(uint32(t>>31)>>1)))
}

//go:noescape
func max32LSX(a *float32, n int, out *[4]int32) int64

//go:noescape
func min32LSX(a *float32, n int, out *[4]int32) int64
