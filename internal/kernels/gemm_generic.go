//go:build !arm64 && !amd64

package kernels

// Generic GEMM micro-kernel for the four targets without a vector-double SIMD
// kernel (loong64, ppc64le, riscv64, s390x): a portable scalar 4x4 register tile
// over the packed panels. There is no hand-vectorized .s here — Go's loong64/
// ppc64le assemblers expose no vector-double arithmetic and riscv64's V extension
// is optional — but the kernel still gets the two structural wins of the packed
// GEMM: the panels are contiguous unit-stride (so the loads are conflict-free,
// the L1-collision fix), and the cache-blocking loop nest plus the multicore
// fan-out (MatMulP) keep it well ahead of an unblocked scalar loop and beat
// single-threaded numpy on large products.

// MR, NR are the register-block dimensions; 4x4 keeps 16 + a few scalars live in
// the GP/FP register file on these targets without spilling.
const (
	MR = 4
	NR = 4
)

// gemmMicro adds the MR x NR tile sum_p pa[p*MR+r]*pb[p*NR+c] into the C block at
// dst[c0 + r*ldc + c], reading the packed (zero-padded) panels. The accumulation
// is the same ikj order as gemmEdge and the scalar oracle, so it is bit-identical
// across all arches. The 16 tile accumulators are explicit locals so the compiler
// keeps them in registers across the kc loop.
func gemmMicro(kc int, pa, pb, dst []float64, ldc int) {
	var (
		c00, c01, c02, c03 float64
		c10, c11, c12, c13 float64
		c20, c21, c22, c23 float64
		c30, c31, c32, c33 float64
	)
	for p := 0; p < kc; p++ {
		a := pa[p*MR : p*MR+4 : p*MR+4]
		b := pb[p*NR : p*NR+4 : p*NR+4]
		a0, a1, a2, a3 := a[0], a[1], a[2], a[3]
		b0, b1, b2, b3 := b[0], b[1], b[2], b[3]
		c00 += a0 * b0
		c01 += a0 * b1
		c02 += a0 * b2
		c03 += a0 * b3
		c10 += a1 * b0
		c11 += a1 * b1
		c12 += a1 * b2
		c13 += a1 * b3
		c20 += a2 * b0
		c21 += a2 * b1
		c22 += a2 * b2
		c23 += a2 * b3
		c30 += a3 * b0
		c31 += a3 * b1
		c32 += a3 * b2
		c33 += a3 * b3
	}
	r0 := dst[0:4:4]
	r0[0] += c00
	r0[1] += c01
	r0[2] += c02
	r0[3] += c03
	r1 := dst[ldc : ldc+4 : ldc+4]
	r1[0] += c10
	r1[1] += c11
	r1[2] += c12
	r1[3] += c13
	r2 := dst[2*ldc : 2*ldc+4 : 2*ldc+4]
	r2[0] += c20
	r2[1] += c21
	r2[2] += c22
	r2[3] += c23
	r3 := dst[3*ldc : 3*ldc+4 : 3*ldc+4]
	r3[0] += c30
	r3[1] += c31
	r3[2] += c32
	r3[3] += c33
}
