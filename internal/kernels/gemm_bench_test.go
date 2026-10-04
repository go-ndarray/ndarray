package kernels

import (
	"strconv"
	"testing"
)

// BenchmarkGemmMicro times the register-blocked micro-kernel alone on
// L1-resident packed panels (kc=256): its GFLOP/s is the per-core ceiling the
// blocked driver can approach, separate from packing and scheduling.
func BenchmarkGemmMicro(b *testing.B) {
	const kc = 256
	pa := make([]float64, kc*MR)
	pb := make([]float64, kc*NR)
	for i := range pa {
		pa[i] = float64(i%7) * 0.5
	}
	for i := range pb {
		pb[i] = float64(i%5) * 0.25
	}
	c := make([]float64, MR*NR)
	b.SetBytes(0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmMicro(kc, pa, pb, c, NR)
	}
	b.ReportMetric(float64(2*MR*NR*kc)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

// BenchmarkMatMulSerial times the whole serial GEMM (packing + blocking +
// micro-kernel, one core) at a few sizes.
func BenchmarkMatMulSerial(b *testing.B) {
	for _, n := range []int{32, 64, 128, 256, 512} {
		a, bb := intMat(n, n, 1), intMat(n, n, 2)
		dst := make([]float64, n*n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clear(dst)
				MatMul(dst, a, bb, n, n, n)
			}
			b.ReportMetric(float64(2*n*n*n)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}
