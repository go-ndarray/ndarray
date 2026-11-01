package benchmarks

import (
	"fmt"
	"testing"

	nd "github.com/go-ndarray/ndarray"
)

// The same reduction and elementwise operation on the float dtypes, which
// have SIMD kernels, and on the integers, which run pure-Go loops, at a
// cache-resident and a memory-bound size. Run them with GOMAXPROCS=1 to
// compare kernels rather than the multicore fan-out.

var dtypeBenchTypes = []nd.DType{nd.Float64, nd.Float32, nd.Int64, nd.Int32}

func BenchmarkDTypeSum(b *testing.B) {
	for _, dt := range dtypeBenchTypes {
		for _, n := range []int{1 << 14, 1 << 20} {
			x := vec(n).AsType(dt)
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				b.SetBytes(int64(n * dt.Size()))
				for i := 0; i < b.N; i++ {
					_ = x.Sum()
				}
			})
		}
	}
}

func BenchmarkDTypeAdd(b *testing.B) {
	for _, dt := range dtypeBenchTypes {
		for _, n := range []int{1 << 14, 1 << 20} {
			x, y := vec(n).AsType(dt), vec(n).AsType(dt)
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				b.SetBytes(int64(n * dt.Size()))
				for i := 0; i < b.N; i++ {
					_, _ = x.Add(y)
				}
			})
		}
	}
}

func BenchmarkDTypeSqrt(b *testing.B) {
	for _, dt := range []nd.DType{nd.Float64, nd.Float32} {
		for _, n := range []int{1 << 14, 1 << 20} {
			x := vec(n).AsType(dt)
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				b.SetBytes(int64(n * dt.Size()))
				for i := 0; i < b.N; i++ {
					_ = x.Sqrt()
				}
			})
		}
	}
}

// BenchmarkDTypeExpLog times Exp and Log on 0.25..24.25 (positive, so Log
// takes its common path; exp stays finite in float32).
func BenchmarkDTypeExpLog(b *testing.B) {
	for _, dt := range []nd.DType{nd.Float64, nd.Float32} {
		for _, n := range []int{1 << 14, 1 << 20} {
			d := make([]float64, n)
			for i := range d {
				d[i] = 0.25 + float64(i%97)*0.25
			}
			x0, _ := nd.FromData(d, n)
			x := x0.AsType(dt)
			for _, f := range []struct {
				name string
				op   func(*nd.Array) *nd.Array
			}{{"Exp", (*nd.Array).Exp}, {"Log", (*nd.Array).Log}} {
				b.Run(fmt.Sprintf("%s/%v/n=%d", f.name, dt, n), func(b *testing.B) {
					b.SetBytes(int64(n * dt.Size()))
					for i := 0; i < b.N; i++ {
						_ = f.op(x)
					}
				})
			}
		}
	}
}

func BenchmarkDTypeMax(b *testing.B) {
	for _, dt := range []nd.DType{nd.Float64, nd.Float32} {
		for _, n := range []int{1 << 14, 1 << 20} {
			x := vec(n).AsType(dt)
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				b.SetBytes(int64(n * dt.Size()))
				for i := 0; i < b.N; i++ {
					_, _ = x.Max()
				}
			})
		}
	}
}

// BenchmarkDTypeMatMul is a square product, 2·n³ flops per op.
func BenchmarkDTypeMatMul(b *testing.B) {
	for _, dt := range []nd.DType{nd.Float64, nd.Float32} {
		for _, n := range []int{64, 256, 1024} {
			x, err := vec(n*n).AsType(dt).Reshape(n, n)
			if err != nil {
				b.Fatal(err)
			}
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if _, err := x.MatMul(x); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkDTypeAddInto is Add without the allocation, the analogue of
// np.add(x, y, out=out): it measures the kernel, where BenchmarkDTypeAdd also
// measures allocating and clearing the result.
func BenchmarkDTypeAddInto(b *testing.B) {
	for _, dt := range dtypeBenchTypes {
		for _, n := range []int{1 << 14, 1 << 20} {
			x, y := vec(n).AsType(dt), vec(n).AsType(dt)
			out := vec(n).AsType(dt)
			b.Run(fmt.Sprintf("%v/n=%d", dt, n), func(b *testing.B) {
				b.SetBytes(int64(n * dt.Size()))
				for i := 0; i < b.N; i++ {
					if err := x.AddInto(out, y); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
