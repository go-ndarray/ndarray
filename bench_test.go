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

// The *Into benchmarks are the no-allocation parity path (np.add(x,y,out=z)):
// they reuse a caller-provided output buffer, so they measure the SIMD kernel
// without the per-op make that NumPy's temp-array cache avoids. This is where Go
// reaches and beats NumPy on small contiguous ops.

func BenchmarkAddInto(b *testing.B) {
	for _, n := range benchElemSizes {
		x, y := benchVec(n), benchVec(n)
		out, _ := New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.AddInto(out, y)
			}
		})
	}
}

func BenchmarkMulInto(b *testing.B) {
	for _, n := range benchElemSizes {
		x, y := benchVec(n), benchVec(n)
		out, _ := New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.MulInto(out, y)
			}
		})
	}
}

func BenchmarkSqrtInto(b *testing.B) {
	for _, n := range benchElemSizes {
		x := benchVec(n)
		out, _ := New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = x.SqrtInto(out)
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

// BenchmarkMatMulVariants covers what BenchmarkMatMul does not: float32
// operands, a transposed operand (read in place by the GEMM) and a strided one
// (a[:, ::2], which the GEMM entry copies first).
func BenchmarkMatMulVariants(b *testing.B) {
	for _, n := range []int{64, 256, 1024} {
		x, y := benchMat(n), benchMat(n)
		x32, y32 := x.AsType(Float32), y.AsType(Float32)
		wide, _ := FromData(make([]float64, 2*n*n), n, 2*n)
		half, _ := wide.Slice(All(), Step(2))
		xt := x.Transpose()
		b.Run(fmt.Sprintf("f32/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = x32.MatMul(y32)
			}
		})
		b.Run(fmt.Sprintf("transA/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = xt.MatMul(y)
			}
		})
		b.Run(fmt.Sprintf("strided/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = half.MatMul(y)
			}
		})
	}
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
