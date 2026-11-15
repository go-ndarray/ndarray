package benchmarks

import "testing"

// BenchmarkUnary times the float64 unary ufuncs on 10^5 elements, the size of
// the single-core table (#118). Round is not routed through a SIMD kernel (it
// rounds half away from zero, which go-simd/floats does not provide); it is
// here as the unchanged control.
func BenchmarkUnary(b *testing.B) {
	const n = 100000
	x := vec(n)
	ops := []struct {
		name string
		do   func()
	}{
		{"Abs", func() { _ = x.Abs() }},
		{"Neg", func() { _ = x.Neg() }},
		{"Square", func() { _ = x.Square() }},
		{"Floor", func() { _ = x.Floor() }},
		{"Ceil", func() { _ = x.Ceil() }},
		{"Round", func() { _ = x.Round() }},
	}
	for _, op := range ops {
		b.Run(op.name, func(b *testing.B) {
			b.SetBytes(n * 8)
			for i := 0; i < b.N; i++ {
				op.do()
			}
		})
	}
}
