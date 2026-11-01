package kernels

// float32 transcendental functions on the float64 kernels.
//
// float32 Exp and Log went through math.Exp and math.Log per element, 6-14x
// slower than NumPy's float32 SIMD on one Zen 3 core. The float64 kernels of
// this package (ports of Arm's optimized-routines, 0.504 ULP worst case in
// float64) are 3-4x faster than math's. A float32 element is widened (exact),
// computed in float64 and rounded once. The result is the correctly rounded
// float32 unless the exact value lies within the float64 kernel's error
// (about half a float64 ULP, 2^-29 of a float32 ULP) of a float32 rounding
// midpoint; it is within 0.5 + 2^-29 float32 ULP everywhere. Overflow and
// underflow follow from the rounding: exp(89) is finite in float64 and
// rounds to +Inf.
//
// The elements go through a 256-element stack buffer, widened, computed in
// place and rounded back, so nothing is allocated. Exp and Log are called
// directly, not through a func value, which would make the buffer escape.

const via64Chunk = 256

type via64Op uint8

const (
	via64Exp via64Op = iota
	via64Log
)

func via64(op via64Op, dst, src []float32) {
	var buf [via64Chunk]float64
	src = src[:len(dst)]
	for i := 0; i < len(dst); i += via64Chunk {
		b := buf[:min(via64Chunk, len(dst)-i)]
		s := src[i : i+len(b)]
		for j := range b {
			b[j] = float64(s[j])
		}
		switch op {
		case via64Exp:
			Exp(b, b)
		case via64Log:
			Log(b, b)
		}
		d := dst[i : i+len(b)]
		for j, v := range b {
			d[j] = float32(v)
		}
	}
}

// Exp32 writes exp(src[i]) into dst[i] for float32 elements.
func Exp32(dst, src []float32) { via64(via64Exp, dst, src) }

// Log32 writes log(src[i]) into dst[i] for float32 elements.
func Log32(dst, src []float32) { via64(via64Log, dst, src) }

// Exp32P is Exp32, split across cores above ParThreshold.
func Exp32P(dst, src []float32) { via64P(via64Exp, dst, src) }

// Log32P is Log32, split across cores above ParThreshold.
func Log32P(dst, src []float32) { via64P(via64Log, dst, src) }

func via64P(op via64Op, dst, src []float32) {
	n := len(dst)
	if n < ParThreshold {
		via64(op, dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { via64(op, dst[lo:hi], src[lo:hi]) })
}
