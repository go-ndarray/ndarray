// Package benchmarks holds the go-ndarray vs NumPy performance-parity harness.
//
// It lives in its own module-internal package (not the library package) so the
// benchmark-only helpers never count against the library's 100%-coverage gate.
// Run it together with docs/bench_numpy.py via gen_report.sh to regenerate
// BENCHMARKS.md.
//
// Every benchmark builds its operands once, then times the op in the loop. The
// shapes, dtype (float64) and data-fill are identical to the NumPy script so the
// two are directly comparable. MatMul covers square and non-square shapes; the
// element-wise / reduction / view ops span small (overhead-relevant) to large
// (bandwidth-bound) sizes.
package benchmarks

import (
	"fmt"
	"testing"

	nd "github.com/go-ndarray/ndarray"
)

// ---- operand builders (deterministic, matched to the numpy script) ----

func vec(n int) *nd.Array {
	d := make([]float64, n)
	for i := range d {
		d[i] = float64(i%97) * 0.5
	}
	a, _ := nd.FromData(d, n)
	return a
}

func mat(r, c int) *nd.Array {
	d := make([]float64, r*c)
	for i := range d {
		d[i] = float64(i%101) * 0.25
	}
	a, _ := nd.FromData(d, r, c)
	return a
}

var elemSizes = []int{1 << 10, 1 << 14, 1 << 18, 1 << 22}

// ---- element-wise ufuncs (alloc form: returns a fresh array) ----

func BenchmarkAdd(b *testing.B) {
	for _, n := range elemSizes {
		x, y := vec(n), vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Add(y)
			}
		})
	}
}

func BenchmarkMul(b *testing.B) {
	for _, n := range elemSizes {
		x, y := vec(n), vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Mul(y)
			}
		})
	}
}

func BenchmarkExp(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.Exp()
			}
		})
	}
}

func BenchmarkSqrt(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.Sqrt()
			}
		})
	}
}

// ---- element-wise *Into (no-allocation parity path, NumPy out=) ----

func BenchmarkAddInto(b *testing.B) {
	for _, n := range elemSizes {
		x, y := vec(n), vec(n)
		out, _ := nd.New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.AddInto(out, y)
			}
		})
	}
}

func BenchmarkMulInto(b *testing.B) {
	for _, n := range elemSizes {
		x, y := vec(n), vec(n)
		out, _ := nd.New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.MulInto(out, y)
			}
		})
	}
}

func BenchmarkSqrtInto(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		out, _ := nd.New(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.SqrtInto(out)
			}
		})
	}
}

// ---- reductions (full) ----

func BenchmarkSum(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_ = x.Sum()
			}
		})
	}
}

func BenchmarkMean(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Mean()
			}
		})
	}
}

func BenchmarkMax(b *testing.B) {
	for _, n := range elemSizes {
		x := vec(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			for i := 0; i < b.N; i++ {
				_, _ = x.Max()
			}
		})
	}
}

// ---- axis reductions (sum/mean/max over a 2-D array) ----

func BenchmarkSumAxis0(b *testing.B) {
	a := mat(1024, 1024)
	b.SetBytes(int64(1024 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = a.SumAxis(0, false)
	}
}

func BenchmarkSumAxis1(b *testing.B) {
	a := mat(1024, 1024)
	b.SetBytes(int64(1024 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = a.SumAxis(1, false)
	}
}

func BenchmarkMaxAxis1(b *testing.B) {
	a := mat(1024, 1024)
	b.SetBytes(int64(1024 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = a.MaxAxis(1, false)
	}
}

// ---- broadcasting (matrix + row vector) ----

func BenchmarkBroadcastAdd(b *testing.B) {
	a := mat(1024, 1024)
	row := mat(1, 1024)
	b.SetBytes(int64(1024 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = a.Add(row)
	}
}

// ---- views / slicing (no-copy stride view) ----

func BenchmarkSliceView(b *testing.B) {
	a := mat(1024, 1024)
	for i := 0; i < b.N; i++ {
		// every other row, columns 100..900 — pure stride arithmetic, no copy.
		_, _ = a.Slice(nd.Rng(0, 1024, 2), nd.R(100, 900))
	}
}

// SliceMaterialize forces the strided view to a contiguous copy (NumPy
// ascontiguousarray): this is the memory-bound fancy-copy comparison.
func BenchmarkSliceMaterialize(b *testing.B) {
	a := mat(1024, 1024)
	v, _ := a.Slice(nd.Rng(0, 1024, 2), nd.R(100, 900))
	b.SetBytes(int64(512 * 800 * 8))
	for i := 0; i < b.N; i++ {
		_ = v.Copy()
	}
}

// ---- concat / stack ----

func BenchmarkConcatAxis0(b *testing.B) {
	x, y := mat(512, 1024), mat(512, 1024)
	b.SetBytes(int64(2 * 512 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = nd.Concatenate([]*nd.Array{x, y}, 0)
	}
}

func BenchmarkStack(b *testing.B) {
	x, y := mat(512, 1024), mat(512, 1024)
	b.SetBytes(int64(2 * 512 * 1024 * 8))
	for i := 0; i < b.N; i++ {
		_, _ = nd.Stack([]*nd.Array{x, y}, 0)
	}
}

// ---- dot / inner / outer ----

func BenchmarkDot1D(b *testing.B) {
	n := 1 << 20
	x, y := vec(n), vec(n)
	b.SetBytes(int64(n * 8))
	for i := 0; i < b.N; i++ {
		_, _ = x.Dot(y)
	}
}

func BenchmarkMatVec(b *testing.B) {
	a := mat(1024, 1024)
	v := vec(1024)
	for i := 0; i < b.N; i++ {
		_, _ = a.Dot(v)
	}
}

func BenchmarkInner(b *testing.B) {
	x, y := mat(512, 512), mat(512, 512)
	for i := 0; i < b.N; i++ {
		_, _ = x.Inner(y)
	}
}

func BenchmarkOuter(b *testing.B) {
	x, y := vec(2048), vec(2048)
	for i := 0; i < b.N; i++ {
		_ = x.Outer(y)
	}
}

// ---- matmul (square + non-square) ----

func BenchmarkMatMul(b *testing.B) {
	for _, n := range []int{32, 64, 128, 256, 512, 1024} {
		x, y := mat(n, n), mat(n, n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = x.MatMul(y)
			}
		})
	}
}

func BenchmarkMatMulSmallRect(b *testing.B) {
	// Small/medium shapes (m x k) @ (k x n) around the serial->parallel GEMM
	// crossover, including the 80²–120² band the 2026-06-23 threshold pass moved
	// onto the parallel path (96x96x96 is the headline gain).
	shapes := [][3]int{{64, 64, 32}, {128, 64, 128}, {96, 96, 96}}
	for _, s := range shapes {
		x, y := mat(s[0], s[1]), mat(s[1], s[2])
		b.Run(fmt.Sprintf("%dx%dx%d", s[0], s[1], s[2]), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = x.MatMul(y)
			}
		})
	}
}

func BenchmarkMatMulNonSquare(b *testing.B) {
	// (1024 x 256) @ (256 x 1024) — a common GEMM aspect ratio.
	x, y := mat(1024, 256), mat(256, 1024)
	for i := 0; i < b.N; i++ {
		_, _ = x.MatMul(y)
	}
}
