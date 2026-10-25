package kernels

import "math"

// The float32 elementwise operations, in the order of the parent package's
// binCode: Bin32 takes these codes.
const (
	Add32 = iota
	Sub32
	Mul32
	Div32
)

// Bin32 writes a[i] OP b[i] into dst[i] for every element, OP one of Add32,
// Sub32, Mul32 or Div32. a and b must be at least as long as dst. A vector
// kernel runs where the architecture has one; each lane is one IEEE float32
// operation, so the result is the scalar loop's bit for bit.
func Bin32(op int, dst, a, b []float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	bin32(op, dst, a[:n], b[:n]) // the kernel reads n of each
}

// bin32Go is the scalar oracle the kernels are tested against, and the
// implementation where there is no kernel.
func bin32Go(op int, dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	switch op {
	case Add32:
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
	case Sub32:
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
	case Mul32:
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
	case Div32:
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
	default:
		panic("kernels: Bin32 op out of range")
	}
}

// Sqrt32 writes the square root of src[i] into dst[i] for every element. src
// must be at least as long as dst. A vector kernel runs where the architecture
// has one; the correctly rounded float32 square root is
// float32(math.Sqrt(float64(x))) bit for bit, since float64 carries more than
// twice float32's precision.
func Sqrt32(dst, src []float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	sqrt32(dst, src[:n]) // the kernel reads n of src
}

// sqrt32Go is the scalar oracle for Sqrt32, and the implementation where
// there is no kernel.
func sqrt32Go(dst, src []float32) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(src[i])))
	}
}

// bin32Func is the shape of the generated float32 kernels: n >= 1 elements.
type bin32Func func(dst, a, b *float32, n int)

// run32 calls k on dst, a and b, all of length n >= 1.
func run32(k bin32Func, dst, a, b []float32) { k(&dst[0], &a[0], &b[0], len(dst)) }
