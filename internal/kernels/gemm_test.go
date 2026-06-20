package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// gemmOracle is the plain triple-loop reference GEMM in ikj order, the exact
// summation order the packed kernel reproduces. dst is pre-zeroed.
func gemmOracle(dst, a, b []float64, m, k, n int) {
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			av := a[i*k+p]
			for j := 0; j < n; j++ {
				dst[i*n+j] += av * b[p*n+j]
			}
		}
	}
}

// withBlocks runs f with the GEMM block sizes forced to (mc,kc,nc) and restores
// them after, so a test can force the multi-block loop nest on small inputs.
func withBlocks(mc, kc, nc int, f func()) {
	omc, okc, onc := blockMC, blockKC, blockNC
	blockMC, blockKC, blockNC = mc, kc, nc
	defer func() { blockMC, blockKC, blockNC = omc, okc, onc }()
	f()
}

// assertGemm checks both MatMul (serial) and MatMulP (parallel forced on) equal
// the oracle bit-for-bit for the given shape, using the integer-valued operands
// that make the ikj order reproduce the oracle exactly.
func assertGemm(t *testing.T, m, k, n int, a, b []float64) {
	t.Helper()
	want := make([]float64, m*n)
	gemmOracle(want, a, b, m, k, n)

	gotS := make([]float64, m*n)
	MatMul(gotS, a, b, m, k, n)
	for i := range want {
		if gotS[i] != want[i] {
			t.Fatalf("MatMul m=%d k=%d n=%d [%d]: %v != %v", m, k, n, i, gotS[i], want[i])
		}
	}
	withThresholds(1<<20, 1, func() { // force the parallel band split
		gotP := make([]float64, m*n)
		MatMulP(gotP, a, b, m, k, n)
		for i := range want {
			if gotP[i] != want[i] {
				t.Fatalf("MatMulP m=%d k=%d n=%d [%d]: %v != %v", m, k, n, i, gotP[i], want[i])
			}
		}
	})
}

// intMat returns an m*k matrix of small integer-valued doubles (exactly
// representable, and their sums of products are exact within float64 for these
// sizes) so the packed kernel must match the oracle bit-for-bit.
func intMat(m, k int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float64, m*k)
	for i := range a {
		a[i] = float64(r.Intn(17) - 8)
	}
	return a
}

// TestGemmCorrectness validates the packed GEMM against the ikj oracle across a
// wide range of shapes: tiny, MR/NR-aligned and ragged (edge tiles on both the
// row and column borders), tall, wide, and sizes that cross several cache
// blocks once the block params are pinned small.
func TestGemmCorrectness(t *testing.T) {
	shapes := []struct{ m, k, n int }{
		{1, 1, 1}, {2, 3, 4}, {4, 4, 4}, {4, 4, 8}, {4, 1, 8},
		{5, 7, 9}, {7, 7, 7}, {8, 8, 8}, {9, 5, 11}, {16, 16, 16},
		{17, 13, 19}, {3, 10, 1}, {1, 10, 3}, {33, 33, 33}, {32, 64, 48},
	}
	for _, s := range shapes {
		a, b := intMat(s.m, s.k, int64(s.m*100+s.k)), intMat(s.k, s.n, int64(s.n*7+s.k))
		// Default blocks.
		assertGemm(t, s.m, s.k, s.n, a, b)
		// Tiny blocks force the NC/KC/MC loop nest and many edge tiles.
		withBlocks(MR, 2, NR, func() { assertGemm(t, s.m, s.k, s.n, a, b) })
		// Asymmetric small blocks.
		withBlocks(2*MR, 3, 2*NR, func() { assertGemm(t, s.m, s.k, s.n, a, b) })
	}
}

// TestGemmFloatTolerance validates the packed GEMM against the oracle for
// genuinely floating-point (non-integer) operands. The packed kernel uses the
// identical ikj summation order, so it is in fact bit-identical here too; we
// assert a tight tolerance to document that no reordering occurs.
func TestGemmFloatTolerance(t *testing.T) {
	r := rand.New(rand.NewSource(99))
	m, k, n := 40, 50, 30
	a := make([]float64, m*k)
	b := make([]float64, k*n)
	for i := range a {
		a[i] = r.NormFloat64()
	}
	for i := range b {
		b[i] = r.NormFloat64()
	}
	want := make([]float64, m*n)
	gemmOracle(want, a, b, m, k, n)
	got := make([]float64, m*n)
	MatMul(got, a, b, m, k, n)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9*(math.Abs(want[i])+1) {
			t.Fatalf("[%d]: %v vs %v", i, got[i], want[i])
		}
	}
}

// TestGemmBufferRealloc forces block sizes larger than the pooled buffers were
// first sized for, exercising the reallocation branch in getPackBuf, then a
// normal-sized call to confirm the pool still serves valid scratch.
func TestGemmBufferRealloc(t *testing.T) {
	// Prime the pool at the default size.
	MatMul(make([]float64, 16), intMat(4, 4, 1), intMat(4, 4, 2), 4, 4, 4)
	// Now demand much larger blocks: getPackBuf must grow both buffers.
	withBlocks(64, 64, 256, func() {
		a, b := intMat(40, 40, 5), intMat(40, 40, 6)
		assertGemm(t, 40, 40, 40, a, b)
	})
	// Back to default blocks; pool buffers are oversized but still valid.
	a, b := intMat(8, 8, 7), intMat(8, 8, 8)
	assertGemm(t, 8, 8, 8, a, b)
}

// TestGemmThresholdPaths exercises the serial (below GemmThreshold) and parallel
// (above) branches of MatMulP explicitly, plus the m==0 and w>m clamps.
func TestGemmThresholdPaths(t *testing.T) {
	a, b := intMat(7, 4, 9), intMat(4, 9, 10)
	want := make([]float64, 7*9)
	gemmOracle(want, a, b, 7, 4, 9)
	check := func(gemm int) {
		withThresholds(1<<14, gemm, func() {
			got := make([]float64, 7*9)
			MatMulP(got, a, b, 7, 4, 9)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("gemm=%d [%d]: %v != %v", gemm, i, got[i], want[i])
				}
			}
		})
	}
	check(1 << 20) // serial path
	check(1)       // parallel path (also w>m for the small row count)

	// m==0 is a no-op and must not panic.
	withThresholds(1<<14, 1, func() {
		MatMulP(nil, nil, b, 0, 4, 9)
	})
}
