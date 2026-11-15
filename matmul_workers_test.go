package ndarray

import (
	"math/rand/v2"
	"runtime"
	"testing"
)

// TestMatMulOneWorkerSameBits is TestGemmOneWorkerSameBits of the GEMM, which
// moved to go-blas/blas with it, asked of MatMul: on non-integer data a product
// above the parallel threshold at GOMAXPROCS=1 (one worker, the single-pass
// path) must equal, bit for bit, the product on four workers, for float64 and,
// where there is a float32 kernel, for float32; with a transposed and a
// decimated operand, which go-blas reads through their strides.
func TestMatMulOneWorkerSameBits(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	random := func(rows, cols int) *Array {
		d := make([]float64, rows*cols)
		for i := range d {
			d[i] = r.NormFloat64()
		}
		a, err := FromData(d, rows, cols)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	for _, s := range [][3]int{{97, 131, 83}, {1, 300, 7000}, {256, 64, 200}} {
		m, k, n := s[0], s[1], s[2]
		wide := random(m, 2*k)
		decimated, err := wide.Slice(All(), Step(2))
		if err != nil {
			t.Fatal(err)
		}
		for name, ab := range map[string][2]*Array{
			"contiguous": {random(m, k), random(k, n)},
			"transposed": {random(k, m).Transpose(), random(n, k).Transpose()},
			"decimated":  {decimated, random(k, n)},
		} {
			dts := []DType{Float64}
			if useGemm32 {
				dts = append(dts, Float32)
			}
			for _, dt := range dts {
				a, b := ab[0].AsType(dt), ab[1].AsType(dt)
				run := func(procs int) *Array {
					defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
					c, err := a.MatMul(b)
					if err != nil {
						t.Fatal(err)
					}
					return c
				}
				one, many := run(1), run(4)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						if one.At(i, j) != many.At(i, j) {
							t.Fatalf("%v %s %v: element (%d,%d) is %v on one worker, %v on four", s, name, dt, i, j, one.At(i, j), many.At(i, j))
						}
					}
				}
			}
		}
	}
}
