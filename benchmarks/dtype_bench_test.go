package benchmarks

import (
	"fmt"
	"testing"

	nd "github.com/go-ndarray/ndarray"
)

// The same reduction and elementwise operation on the dtypes that have SIMD
// kernels (float64) and on those that run the pure-Go loops (float32 and the
// integers), at a cache-resident and a memory-bound size. Run them with
// GOMAXPROCS=1 to compare kernels rather than the multicore fan-out. They
// measure what a float32 kernel would be worth before anyone writes one.

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
