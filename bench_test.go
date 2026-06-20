package ndarray

import (
	"fmt"
	"testing"
)

// Benchmarks for the operations compared against numpy in docs/perf.md. Sizes
// span small (threshold-relevant) to large (where multicore + SIMD win). Each
// benchmark builds its operands once and times the operation in the loop.

var benchElemSizes = []int{1 << 10, 1 << 14, 1 << 18, 1 << 22}

func benchVec(n int) *Array {
	d := make([]float64, n)
	for i := range d {
		d[i] = float64(i%97) * 0.5
	}
	a, _ := FromData(d, n)
	return a
}

func BenchmarkAdd(b *testing.B) {
	for _, n := range benchElemSizes {
		x, y := benchVec(n), benchVec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Add(y)
			}
		})
	}
}

func BenchmarkMul(b *testing.B) {
	for _, n := range benchElemSizes {
		x, y := benchVec(n), benchVec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Mul(y)
			}
		})
	}
}

func BenchmarkSqrt(b *testing.B) {
	for _, n := range benchElemSizes {
		x := benchVec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = x.Sqrt()
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	for _, n := range benchElemSizes {
		x := benchVec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = x.Sum()
			}
		})
	}
}

func BenchmarkMax(b *testing.B) {
	for _, n := range benchElemSizes {
		x := benchVec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = x.Max()
			}
		})
	}
}

func benchMat(n int) *Array {
	d := make([]float64, n*n)
	for i := range d {
		d[i] = float64(i%101) * 0.25
	}
	a, _ := FromData(d, n, n)
	return a
}

func BenchmarkMatMul(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512, 1024} {
		x, y := benchMat(n), benchMat(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = x.MatMul(y)
			}
		})
	}
}
