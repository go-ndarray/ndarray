package kernels

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// withThresholds runs f with the parallel thresholds forced and restores them
// after, so a test can exercise both the serial (below-threshold) and the
// parallel (above-threshold) branches deterministically. The gemm argument pins
// matVecThreshold (MatVecP), so a single low/high value flips the mat-vec
// kernel onto the serial or parallel path; ParThreshold drives the
// elementwise/reduction kernels. (The GEMM has its own threshold, in go-blas.)
func withThresholds(par, gemm int, f func()) {
	op, omv := ParThreshold, matVecThreshold
	ParThreshold, matVecThreshold = par, gemm
	defer func() { ParThreshold, matVecThreshold = op, omv }()
	f()
}

// withMaxProcs runs f with GOMAXPROCS forced to n and restores it after. The
// worker count of the parallel kernels is derived from GOMAXPROCS, so a test
// that must hit the `workers > extent` clamp (e.g. more workers than output
// rows) pins it high here — otherwise the clamp branch is unreachable on a
// low-core CI runner and the 100% coverage gate fails there but not locally.
func withMaxProcs(n int, f func()) {
	old := runtime.GOMAXPROCS(n)
	defer runtime.GOMAXPROCS(old)
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

// TestMapReduceManyWorkers pins GOMAXPROCS far above the element count so the
// worker count w exceeds the number of non-empty chunks: with chunk =
// ceil(n/w), the stride idx*chunk overruns n before the last worker, so a naive
// loop indexes a[s:e] with s >= n and panics ("slice bounds out of range
// [>n:n]") — the exact wide-SMP crash seen on a 160-core POWER box. The fix
// drops the empty workers, so SumP/MaxP/MinP must run cleanly and stay
// bit-identical (Max/Min) or within tolerance (Sum) to the serial reducers for
// any worker:element ratio. Sizes are chosen so n < w (more workers than
// elements) and n just above the forced threshold.
func TestMapReduceManyWorkers(t *testing.T) {
	withMaxProcs(256, func() {
		for _, n := range []int{5, 7, 8, 9, 16, 33, 100, 257, 1000} {
			a := randVec(n, 11)
			withThresholds(1, 1<<14, func() { // force parallel path even for n=5
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
	})
}

// TestRunAxisP checks the parallel axis-reduction driver reproduces the serial
// SumAxis/MaxAxis kernel exactly, across the serial (below-threshold), the
// outer-split parallel, and the single-outer-slab (outer<2, axis-0) paths.
func TestRunAxisP(t *testing.T) {
	shapes := []struct{ outer, axisLen, inner int }{
		{1, 5, 7},   // single outer slab -> serial path inside RunAxisP
		{6, 4, 3},   // multi outer
		{64, 8, 2},  // larger, exercises the worker split when threshold is low
		{100, 1, 1}, // axisLen 1
	}
	for _, s := range shapes {
		n := s.outer * s.axisLen * s.inner
		src := randVec(n, int64(n))
		for _, par := range []int{1 << 20, 4} { // serial, then parallel
			run := func() {
				withThresholds(par, 1<<14, func() {
					for _, kk := range []axisKernel{SumAxis, MaxAxis} {
						gotD := make([]float64, s.outer*s.inner)
						wantD := make([]float64, s.outer*s.inner)
						RunAxisP(kk, gotD, src, s.outer, s.axisLen, s.inner)
						kk(wantD, src, s.outer, s.axisLen, s.inner, 0, s.inner)
						for i := range wantD {
							if gotD[i] != wantD[i] {
								t.Fatalf("RunAxisP %v par=%d [%d]: %v != %v", s, par, i, gotD[i], wantD[i])
							}
						}
					}
				})
			}
			// Pin GOMAXPROCS above the outer extents so the worker>outer clamp
			// fires regardless of the runner's core count.
			withMaxProcs(128, run)
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

// naiveAxis is an independent oracle for the axis reductions: a plain triple
// loop combining the middle axis in order with op.
func naiveAxis(src []float64, outer, axisLen, inner int, op func(acc, v float64) float64) []float64 {
	out := make([]float64, outer*inner)
	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			acc := src[o*axisLen*inner+i]
			for k := 1; k < axisLen; k++ {
				acc = op(acc, src[(o*axisLen+k)*inner+i])
			}
			out[o*inner+i] = acc
		}
	}
	return out
}

// TestRunAxisPBands drives every RunAxisP schedule — serial, outer split,
// column split (one slab, and fewer slabs than workers) and the narrow
// fallback — through Sum/Prod/Max/Min against the naive in-order oracle. With
// inner > 1 the band kernels combine rows in axis order (through the SIMD
// elementwise kernel from rowRun columns on), so the result must be exact; the
// inner == 1 row sum is regrouped by the SIMD sum and checked to 1e-12.
func TestRunAxisPBands(t *testing.T) {
	type op struct {
		name string
		k    axisKernel
		f    func(a, v float64) float64
	}
	ops := []op{
		{"sum", SumAxis, func(a, v float64) float64 { return a + v }},
		{"prod", ProdAxis, func(a, v float64) float64 { return a * v }},
		{"max", MaxAxis, func(a, v float64) float64 { return max(a, v) }},
		{"min", MinAxis, func(a, v float64) float64 { return min(a, v) }},
	}
	shapes := []struct{ outer, axisLen, inner int }{
		{1, 6, 3 * axisBand},    // one slab: column bands
		{3, 5, 2*axisBand + 37}, // fewer slabs than workers: ragged column bands
		{3, 4, 100},             // fewer slabs, too narrow to band: outer fallback
		{40, 3, rowRun + 5},     // outer split, SIMD row combine
		{2, 7, rowRun - 1},      // scalar row combine
		{50, 300, 1},            // row sums (inner == 1)
	}
	for _, s := range shapes {
		src := randVec(s.outer*s.axisLen*s.inner, int64(s.inner))
		for i := range src {
			src[i] = 0.5 + src[i]*0.25 // keep products finite
		}
		for _, par := range []int{1 << 30, 1} {
			withMaxProcs(8, func() {
				withThresholds(par, 1<<14, func() {
					for _, o := range ops {
						want := naiveAxis(src, s.outer, s.axisLen, s.inner, o.f)
						got := make([]float64, len(want))
						RunAxisP(o.k, got, src, s.outer, s.axisLen, s.inner)
						for i := range want {
							exact := s.inner > 1 || o.name != "sum"
							if exact && got[i] != want[i] ||
								!exact && math.Abs(got[i]-want[i]) > 1e-12*math.Abs(want[i]) {
								t.Fatalf("%s %v par=%d [%d]: %v, want %v", o.name, s, par, i, got[i], want[i])
							}
						}
					}
				})
			})
		}
	}
}
