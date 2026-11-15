package benchmarks

import (
	"testing"

	nd "github.com/go-ndarray/ndarray"
)

// BenchmarkMaxMin times the float64 pairwise Maximum and Minimum on 10^5
// elements, the size of the single-core table (#118), including the result
// allocation.
func BenchmarkMaxMin(b *testing.B) {
	const n = 100000
	x := vec(n)
	d := make([]float64, n)
	for i := range d {
		d[i] = float64((i*31)%89) * 0.5 // differs from vec(n), so both operands win in turn
	}
	y, _ := nd.FromData(d, n)
	ops := []struct {
		name string
		do   func()
	}{
		{"Maximum", func() { _, _ = x.Maximum(y) }},
		{"Minimum", func() { _, _ = x.Minimum(y) }},
	}
	for _, op := range ops {
		b.Run(op.name, func(b *testing.B) {
			b.SetBytes(2 * n * 8)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				op.do()
			}
		})
	}
}
