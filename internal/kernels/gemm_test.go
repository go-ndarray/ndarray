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

// withBandTiles runs f with the MatMulP dynamic-band height (in MR-tiles) forced
// to bt and restores it after, so a test can drive the band-sizing clamps.
func withBandTiles(bt int, f func()) {
	o := bandTilesMR
	bandTilesMR = bt
	defer func() { bandTilesMR = o }()
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
		// Block dims that are NOT multiples of MR/NR: the bottom A panel and the
		// right B panel of each block are then partial and must be zero-padded to
		// a full MR/NR tile in scratch sized via roundUp — this is the regression
		// guard for the buffer-overflow bug a non-MR-aligned blockMC exposed.
		withBlocks(MR+1, 2, NR+1, func() { assertGemm(t, s.m, s.k, s.n, a, b) })
		withBlocks(3*MR-1, 5, 3*NR-1, func() { assertGemm(t, s.m, s.k, s.n, a, b) })
	}
	// A size large enough that the DEFAULT (256/256/512) blocks apply with the
	// real MR=6 not dividing blockMC=256 — the production configuration that
	// panicked before the roundUp fix. Crosses the parallel band split too.
	withThresholds(1<<14, 1, func() {
		m, k, n := 300, 200, 260
		a, b := intMat(m, k, 1), intMat(k, n, 2)
		assertGemm(t, m, k, n, a, b)
	})
}

// TestGemmBandSizing drives the two band-sizing clamps in MatMulP's parallel
// path against the oracle: (a) the small-m shrink (band wider than m/(4w) rounds
// down so every worker still gets a band), and (b) the band > blockMC cap (a
// band taller than the MC row block is clamped to MC so each A panel still fits).
// Both must leave the result bit-identical to the serial GEMM.
func TestGemmBandSizing(t *testing.T) {
	// (a) Small-m shrink: tiny row count, default band height would exceed
	// m/(4w); the want<bandRows branch fires and shrinks the band.
	withMaxProcs(8, func() {
		withThresholds(1<<14, 1, func() { // force parallel
			a, b := intMat(10, 7, 21), intMat(7, 9, 22)
			assertGemm(t, 10, 7, 9, a, b)
		})
	})
	// (b) band > blockMC cap: pin blockMC below the band height and feed enough
	// rows that the m/(4w) shrink does NOT pre-empt the cap (want >= bandRows),
	// so the bandRows>blockMC clamp is the one that fires.
	withMaxProcs(2, func() {
		withBandTiles(2, func() { // bandRows = 2*MR
			withBlocks(MR, 3, 2*NR, func() { // blockMC = MR < bandRows
				withThresholds(1<<14, 1, func() {
					m, k, n := 40*MR, 5, 2*NR
					a, b := intMat(m, k, 23), intMat(k, n, 24)
					assertGemm(t, m, k, n, a, b)
				})
			})
		})
	})
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

// TestDot1D checks the exact left-to-right scalar dot oracle (the reference the
// unrolled/parallel variants document their ULP trade-off against).
func TestDot1D(t *testing.T) {
	for _, n := range []int{0, 1, 4, 9, 100} {
		a, b := randVec(n, int64(7*n+1)), randVec(n, int64(7*n+2))
		var want float64
		for i := range a {
			want += a[i] * b[i]
		}
		if got := Dot1D(a, b); got != want {
			t.Fatalf("Dot1D n=%d: %v != %v", n, got, want)
		}
	}
}

// TestDotRange checks the unrolled contiguous dot against a plain reference for
// lengths that exercise the 4-wide body and every remainder (0..3 trailing).
func TestDotRange(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 13, 1000} {
		a, b := randVec(n, int64(n+1)), randVec(n, int64(n+2))
		var want float64
		for i := range a {
			want += a[i] * b[i]
		}
		got := dotRange(a, b)
		if math.Abs(got-want) > 1e-9*(1+math.Abs(want)) {
			t.Fatalf("dotRange n=%d: %v != %v", n, got, want)
		}
	}
}

// TestMatVecP checks mat·vec over both the serial (below GemmThreshold) and
// parallel (above, plus the w>m clamp) paths against a row-dot reference.
func TestMatVecP(t *testing.T) {
	m, k := 7, 13
	a, v := randVec(m*k, 3), randVec(k, 4)
	want := make([]float64, m)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			want[i] += a[i*k+p] * v[p]
		}
	}
	check := func(gemm int) {
		withThresholds(1<<14, gemm, func() {
			got := make([]float64, m)
			MatVecP(got, a, v, m, k)
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-9*(1+math.Abs(want[i])) {
					t.Fatalf("MatVecP gemm=%d [%d]: %v != %v", gemm, i, got[i], want[i])
				}
			}
		})
	}
	check(1 << 20) // serial
	// Pin GOMAXPROCS above m so the worker>m clamp fires on any-core CI.
	withMaxProcs(128, func() { check(1) }) // parallel + w>m clamp
}

// TestVecMatP checks vec·mat (1-D · 2-D) against a column-accumulation reference.
func TestVecMatP(t *testing.T) {
	k, n := 11, 9
	v, a := randVec(k, 5), randVec(k*n, 6)
	want := make([]float64, n)
	for p := 0; p < k; p++ {
		for j := 0; j < n; j++ {
			want[j] += v[p] * a[p*n+j]
		}
	}
	got := make([]float64, n)
	VecMatP(got, v, a, k, n)
	for j := range want {
		if math.Abs(got[j]-want[j]) > 1e-9*(1+math.Abs(want[j])) {
			t.Fatalf("VecMatP [%d]: %v != %v", j, got[j], want[j])
		}
	}
}

// TestDot1DP checks the parallel 1-D dot below and above ParThreshold against a
// straight reference sum.
func TestDot1DP(t *testing.T) {
	for _, n := range []int{1, 1000, 100000} {
		a, b := randVec(n, int64(n)), randVec(n, int64(2*n))
		var want float64
		for i := range a {
			want += a[i] * b[i]
		}
		for _, par := range []int{1 << 20, 4} { // serial, then parallel
			withMaxProcs(128, func() {
				withThresholds(par, 1<<14, func() {
					got := Dot1DP(a, b)
					if math.Abs(got-want) > 1e-6*(1+math.Abs(want)) {
						t.Fatalf("Dot1DP n=%d par=%d: %v != %v", n, par, got, want)
					}
				})
			})
		}
	}
}
