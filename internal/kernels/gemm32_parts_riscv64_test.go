package kernels

import (
	"fmt"
	"testing"
)

// BenchmarkGemm32Parts times the pieces of a float32 n x n product on the
// current kernel: the whole MatMul32StridedP, packing A, packing B, and the
// micro-kernel tiles alone (packed panels, no packing).
func BenchmarkGemm32Parts(b *testing.B) {
	if gemm32 == nil {
		b.Skip("no float32 GEMM kernel here")
	}
	g := gemm32
	for _, n := range []int{64, 256} {
		a := rowMajor(intMat32(n, n, 1), n)
		bm := rowMajor(intMat32(n, n, 2), n)
		dst := make([]float32, n*n)
		kc := min(n, blockKC)
		pa := make([]float32, g.paCap())
		pb := make([]float32, g.pbCap())
		b.Run(fmt.Sprintf("full/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clear(dst)
				MatMul32StridedP(dst, a, bm, n, n, n)
			}
		})
		var sink []float32
		b.Run(fmt.Sprintf("alloc/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink = make([]float32, n*n)
			}
		})
		_ = sink
		b.Run(fmt.Sprintf("clear/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clear(dst)
			}
		})
		b.Run(fmt.Sprintf("packA/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				g.packA(a, 0, min(n, blockMC), 0, kc, pa)
			}
		})
		b.Run(fmt.Sprintf("packB/n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				g.packB(bm, 0, kc, 0, min(n, blockNC), pb)
			}
		})
		b.Run(fmt.Sprintf("micro/n=%d", n), func(b *testing.B) {
			mc, nc := min(n, blockMC), min(n, blockNC)
			for i := 0; i < b.N; i++ {
				for jr := 0; jr+g.nr <= nc; jr += g.nr {
					for ir := 0; ir+g.mr <= mc; ir += g.mr {
						g.micro(kc, pa[ir*kc:], pb[jr*kc:], dst[ir*n+jr:], n)
					}
				}
			}
		})
	}
}
