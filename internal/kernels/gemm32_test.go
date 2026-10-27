package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// gemmOracle32 is gemmOracle in float32 arithmetic: the ikj order, each product
// and sum rounded to float32.
func gemmOracle32(dst []float32, a, b MatOf[float32], m, k, n int) {
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			av := a.Data[a.Off+i*a.RS+p*a.CS]
			for j := 0; j < n; j++ {
				dst[i*n+j] += av * b.Data[b.Off+p*b.RS+j*b.CS]
			}
		}
	}
}

func intMat32(m, k int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(r.Intn(17) - 8)
	}
	return a
}

// assertGemm32 checks the float32 GEMM, serial and with the parallel band split
// forced, against the oracle bit for bit on integer-valued operands, whose
// products and sums are exact in float32 at these sizes.
func assertGemm32(t *testing.T, m, k, n int, a, b Mat32) {
	t.Helper()
	want := make([]float32, m*n)
	gemmOracle32(want, a, b, m, k, n)
	saved := unpackA
	defer func() { unpackA = saved }()
	for _, unpacked := range []bool{true, false} {
		unpackA = unpacked
		for _, gemm := range []int{1 << 30, 1} {
			withThresholds(1<<20, gemm, func() {
				got := make([]float32, m*n)
				MatMul32StridedP(got, a, b, m, k, n)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("m=%d k=%d n=%d gemm=%d unpackA=%v [%d]: %v != %v",
							m, k, n, gemm, unpacked, i, got[i], want[i])
					}
				}
			})
		}
	}
}

// TestGemm32Correctness is TestGemmCorrectness for float32: every shape and
// block configuration that drives the edge tiles and the loop nest, on
// row-major operands and on a transposed, offset B view.
func TestGemm32Correctness(t *testing.T) {
	g := gemm32
	if !HasGemm32() {
		t.Skip("no float32 GEMM kernel here")
	}
	MR, NR := g.mr, g.nr
	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {6, 4, 16}, {8, 8, 8}, {5, 7, 9}, {7, 7, 17},
		{9, 5, 11}, {16, 16, 16}, {17, 13, 19}, {3, 10, 1}, {1, 10, 3},
		{33, 33, 33}, {32, 64, 48},
	}
	for _, s := range shapes {
		a := rowMajor(intMat32(s.m, s.k, int64(s.m*100+s.k)), s.k)
		b := rowMajor(intMat32(s.k, s.n, int64(s.n*7+s.k)), s.n)
		// bt holds B transposed behind a 3-element offset; the view reads it
		// back as B, through the packB column walk (CS != 1).
		bt := Mat32{Data: make([]float32, 3+s.k*s.n), Off: 3, RS: 1, CS: s.k}
		for p := 0; p < s.k; p++ {
			for j := 0; j < s.n; j++ {
				bt.Data[3+j*s.k+p] = b.Data[p*s.n+j]
			}
		}
		// as holds A behind a 2-element offset, rows k+3 apart: the
		// unpacked-A kernels read it through lda != k.
		as := Mat32{Data: make([]float32, 2+s.m*(s.k+3)), Off: 2, RS: s.k + 3, CS: 1}
		for i := 0; i < s.m; i++ {
			copy(as.Data[2+i*(s.k+3):], a.Data[i*s.k:(i+1)*s.k])
		}
		assertGemm32(t, s.m, s.k, s.n, as, b)
		for _, bv := range []Mat32{b, bt} {
			assertGemm32(t, s.m, s.k, s.n, a, bv)
			withBlocks(MR, 2, NR, func() { assertGemm32(t, s.m, s.k, s.n, a, bv) })
			withBlocks(2*MR, 3, 2*NR, func() { assertGemm32(t, s.m, s.k, s.n, a, bv) })
			withBlocks(MR+1, 2, NR+1, func() { assertGemm32(t, s.m, s.k, s.n, a, bv) })
			withBlocks(3*MR-1, 5, 3*NR-1, func() { assertGemm32(t, s.m, s.k, s.n, a, bv) })
		}
	}
	// Default blocks with MR not dividing blockMC, across the band split.
	m, k, n := 300, 200, 260
	assertGemm32(t, m, k, n, rowMajor(intMat32(m, k, 1), k), rowMajor(intMat32(k, n, 2), n))
}

// TestGemm32Accuracy bounds the float32 GEMM on random data by the classical
// error bound of a float32 dot product, k*eps32*sum|a||b|, against the product
// computed in float64.
func TestGemm32Accuracy(t *testing.T) {
	if gemm32 == nil {
		t.Skip("no float32 GEMM kernel here")
	}
	r := rand.New(rand.NewSource(5))
	m, k, n := 70, 300, 50
	a, b := make([]float32, m*k), make([]float32, k*n)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	for i := range b {
		b[i] = float32(r.NormFloat64())
	}
	got := make([]float32, m*n)
	MatMul32StridedP(got, rowMajor(a, k), rowMajor(b, n), m, k, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var exact, abs float64
			for p := 0; p < k; p++ {
				x := float64(a[i*k+p]) * float64(b[p*n+j])
				exact += x
				abs += math.Abs(x)
			}
			if d := math.Abs(float64(got[i*n+j]) - exact); d > float64(k)*0x1p-24*abs {
				t.Fatalf("[%d,%d]: %v vs %v (off %g, bound %g)", i, j, got[i*n+j], exact, d, float64(k)*0x1p-24*abs)
			}
		}
	}
}

// TestGemm32Specials: NaN and Inf in an operand reach exactly the outputs
// whose dot products they enter.
func TestGemm32Specials(t *testing.T) {
	if gemm32 == nil {
		t.Skip("no float32 GEMM kernel here")
	}
	m, k, n := 13, 9, 21
	for _, special := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		a, b := intMat32(m, k, 3), intMat32(k, n, 4)
		for i := range b {
			if b[i] == 0 {
				b[i] = 1 // keep Inf*0 out: the row of a meets every b entry
			}
		}
		a[4*k+2] = special // row 4
		got := make([]float32, m*n)
		MatMul32StridedP(got, rowMajor(a, k), rowMajor(b, n), m, k, n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				v := float64(got[i*n+j])
				bad := math.IsNaN(v) || math.IsInf(v, 0)
				if bad != (i == 4) {
					t.Fatalf("%v at row 4: [%d,%d] = %v", special, i, j, v)
				}
			}
		}
	}
}
