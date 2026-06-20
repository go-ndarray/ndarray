package kernels

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// withThresholds runs f with the parallel thresholds forced to v and restores
// them after, so a test can exercise both the serial (below-threshold) and the
// parallel (above-threshold) branches deterministically.
func withThresholds(par, gemm int, f func()) {
	op, og := ParThreshold, GemmThreshold
	ParThreshold, GemmThreshold = par, gemm
	defer func() { ParThreshold, GemmThreshold = op, og }()
	f()
}

func randVec(n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float64, n)
	for i := range a {
		a[i] = r.NormFloat64() * 100
	}
	return a
}

// TestParallelElementwise checks every parallel elementwise kernel produces the
// exact serial result, both below and above the (forced-low) threshold.
func TestParallelElementwise(t *testing.T) {
	for _, n := range []int{1, 5, 1000, 100000} {
		a, b := randVec(n, 1), randVec(n, 2)
		for _, par := range []int{1 << 20, 4} { // serial path, then parallel path
			withThresholds(par, 1<<14, func() {
				check := func(name string, p, s func(dst, a, b []float64)) {
					gotP, gotS := make([]float64, n), make([]float64, n)
					p(gotP, a, b)
					s(gotS, a, b)
					for i := range gotP {
						if gotP[i] != gotS[i] {
							t.Fatalf("%s n=%d par=%d [%d]: %v != %v", name, n, par, i, gotP[i], gotS[i])
						}
					}
				}
				check("Add", AddP, Add)
				check("Sub", SubP, Sub)
				check("Mul", MulP, Mul)
				check("Div", DivP, Div)

				gotP, gotS := make([]float64, n), make([]float64, n)
				sq := func(x float64) float64 { return x * x }
				MapP(gotP, a, sq)
				Map(gotS, a, sq)
				for i := range gotP {
					if gotP[i] != gotS[i] {
						t.Fatalf("Map n=%d par=%d [%d]: %v != %v", n, par, i, gotP[i], gotS[i])
					}
				}
			})
		}
	}
}

// TestParallelReductions checks SumP (to tolerance, grouping differs) and
// MaxP/MinP (exact, associative) against the serial reducers, parallel path on.
func TestParallelReductions(t *testing.T) {
	for _, n := range []int{1, 7, 1000, 100000} {
		a := randVec(n, 3)
		withThresholds(4, 1<<14, func() {
			gotSum, wantSum := SumP(a), Sum(a)
			tol := 1e-9 * (float64(n) + 1) * (math.Abs(wantSum) + 1)
			if math.Abs(gotSum-wantSum) > tol {
				t.Fatalf("SumP n=%d: %v vs %v (tol %g)", n, gotSum, wantSum, tol)
			}
			if got, want := MaxP(a), Max(a); got != want {
				t.Fatalf("MaxP n=%d: %v != %v", n, got, want)
			}
			if got, want := MinP(a), Min(a); got != want {
				t.Fatalf("MinP n=%d: %v != %v", n, got, want)
			}
		})
	}
}

// TestParallelMatMul checks MatMulP equals the serial MatMul for non-square
// shapes, both below and above the (forced-low) GEMM threshold.
func TestParallelMatMul(t *testing.T) {
	// The {2,..} case with GOMAXPROCS>2 and the parallel threshold forced to 1
	// exercises the w>m clamp (more cores than output rows). {0,..} hits the
	// m==0 path; the rest cover assorted non-square bands.
	cases := []struct{ m, k, n int }{{0, 3, 4}, {1, 1, 1}, {2, 6, 8}, {3, 5, 2}, {7, 4, 9}, {64, 48, 40}}
	for _, c := range cases {
		a, b := randVec(c.m*c.k, 10), randVec(c.k*c.n, 11)
		for _, gemm := range []int{1 << 20, 1} { // serial, then parallel
			withThresholds(1<<14, gemm, func() {
				gotP := make([]float64, c.m*c.n)
				gotS := make([]float64, c.m*c.n)
				MatMulP(gotP, a, b, c.m, c.k, c.n)
				MatMul(gotS, a, b, c.m, c.k, c.n)
				for i := range gotP {
					if gotP[i] != gotS[i] {
						t.Fatalf("MatMulP %v gemm=%d [%d]: %v != %v", c, gemm, i, gotP[i], gotS[i])
					}
				}
			})
		}
	}
}

// TestSqrtP checks the parallel sqrt wrapper is bit-identical to a scalar
// math.Sqrt loop on both the serial (below-threshold) and parallel branches.
func TestSqrtP(t *testing.T) {
	for _, n := range []int{1, 7, 1000, 100000} {
		src := randVec(n, 21)
		for i := range src {
			if src[i] < 0 {
				src[i] = -src[i] // keep most finite & non-negative; NaN/Inf are covered in TestSqrtSIMD
			}
		}
		for _, par := range []int{1 << 20, 4} { // serial path, then parallel path
			withThresholds(par, 1<<14, func() {
				gotP := make([]float64, n)
				want := make([]float64, n)
				SqrtP(gotP, src)
				sqrtScalar(want, src)
				for i := range gotP {
					if !bitEq(gotP[i], want[i]) {
						t.Fatalf("SqrtP n=%d par=%d [%d]: %v != %v", n, par, i, gotP[i], want[i])
					}
				}
			})
		}
	}
}

// TestMatMulBlockDispatch exercises BOTH MatMul paths — the register-blocked
// micro-kernel (n <= blockMaxN) and the autovectorized ikj kernel (n >
// blockMaxN) — by pinning blockMaxN, and asserts they produce identical results
// (the dispatch is a pure speed choice). It also covers the blocked kernel's
// row-remainder tail (m not a multiple of 4).
func TestMatMulBlockDispatch(t *testing.T) {
	ob := blockMaxN
	defer func() { blockMaxN = ob }()
	cases := []struct{ m, k, n int }{{4, 3, 5}, {6, 4, 5}, {7, 5, 8}, {8, 8, 16}}
	for _, c := range cases {
		a, b := randVec(c.m*c.k, 30), randVec(c.k*c.n, 31)
		blocked := make([]float64, c.m*c.n)
		ikj := make([]float64, c.m*c.n)
		blockMaxN = 1 << 30 // force the blocked path
		MatMul(blocked, a, b, c.m, c.k, c.n)
		blockMaxN = 0 // force the ikj path
		MatMul(ikj, a, b, c.m, c.k, c.n)
		for i := range blocked {
			if blocked[i] != ikj[i] {
				t.Fatalf("blocked vs ikj %v [%d]: %v != %v", c, i, blocked[i], ikj[i])
			}
		}
	}
}

// TestNumWorkers exercises the worker-count clamps (GOMAXPROCS bound, the
// one-worker-per-slab cap, and the floor of 1).
func TestNumWorkers(t *testing.T) {
	old := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(old)
	runtime.GOMAXPROCS(4)
	withThresholds(1000, 1<<14, func() {
		// n just over one slab -> capped to 1 worker by the work bound.
		if w := numWorkers(1001); w != 2 {
			t.Errorf("numWorkers(1001) = %d, want 2", w)
		}
		// Huge n -> capped to GOMAXPROCS.
		if w := numWorkers(1 << 20); w != 4 {
			t.Errorf("numWorkers(1<<20) = %d, want 4", w)
		}
	})
	// GOMAXPROCS forced to 1 -> single worker.
	runtime.GOMAXPROCS(1)
	withThresholds(4, 1<<14, func() {
		if w := numWorkers(1000); w != 1 {
			t.Errorf("numWorkers with GOMAXPROCS=1 = %d, want 1", w)
		}
		// parallelFor with w==1 takes the serial body path.
		got := make([]float64, 8)
		parallelFor(8, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				got[i] = float64(i)
			}
		})
		for i := range got {
			if got[i] != float64(i) {
				t.Fatalf("parallelFor serial [%d] = %v", i, got[i])
			}
		}
	})
}
