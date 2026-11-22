package kernels

import "github.com/go-simd/floats"

// bin32 runs go-simd/floats Add32/Sub32/Mul32/Div32 for op. dst, a and b have
// the same length n >= 1 (Bin32 reslices them), which is what floats requires:
// it processes len(a) elements and panics on a shorter operand.
func bin32(op int, dst, a, b []float32) {
	switch op {
	case Add32:
		floats.Add32(dst, a, b)
	case Sub32:
		floats.Sub32(dst, a, b)
	case Mul32:
		floats.Mul32(dst, a, b)
	case Div32:
		floats.Div32(dst, a, b)
	default:
		panic("kernels: Bin32 op out of range")
	}
}

// sqrt32 runs go-simd/floats Sqrt32; dst and src have the same length n >= 1
// (Sqrt32 reslices src).
func sqrt32(dst, src []float32) { floats.Sqrt32(dst, src) }
