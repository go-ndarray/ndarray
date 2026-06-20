package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestSumSIMD asserts the SIMD sum kernel agrees with the scalar oracle to a
// tight relative tolerance across many random lengths (covering the vector body,
// the lane fold, and the scalar tail of every residue mod 8). The two use a
// different but equally valid summation order — lane-parallel vs strictly
// sequential — so they are NOT required to be bit-identical, only close, the
// same contract numpy's pairwise summation makes. On architectures without a
// SIMD kernel sumSIMD aliases the scalar Sum, so this trivially holds and still
// exercises the dispatch.
func TestSumSIMD(t *testing.T) {
	t.Logf("HaveReduceSIMD = %v (true: validating a real SIMD kernel; "+
		"false: scalar alias on this arch)", HaveReduceSIMD)
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 100, 1000, 4096, 9999} {
		a := make([]float64, n)
		for i := range a {
			a[i] = r.NormFloat64() * 1e3
		}
		got := sumSIMD(a)
		want := Sum(a)
		if n == 0 {
			if got != 0 {
				t.Fatalf("n=0: sumSIMD = %v, want 0", got)
			}
			continue
		}
		// Relative tolerance scaled by n (accumulated rounding grows with n).
		tol := 1e-12 * (float64(n) + 1) * (math.Abs(want) + 1)
		if math.Abs(got-want) > tol {
			t.Fatalf("n=%d: sumSIMD = %.17g, scalar = %.17g, |diff|=%g > tol=%g",
				n, got, want, math.Abs(got-want), tol)
		}
	}
}

// TestBinSIMD asserts the SIMD elementwise binary kernels (addBin/subBin/mulBin/
// divBin) are BIT-IDENTICAL to the scalar Add/Sub/Mul/Div oracle across many
// lengths (vector body + scalar tail of every residue mod 8) and across IEEE
// edge inputs (signed zeros, infinities, NaN, extremes). Each lane is a single,
// independent, correctly-rounded IEEE op with no grouping freedom, so bit-
// identity (not mere closeness) is the contract — including on arm64, where add/
// sub/mul are computed via an FMA against an exact constant (b+a*1.0, a-b*1.0,
// 0+a*b), which rounds identically to the plain op. On the arches without a
// vector-double kernel the *Bin functions alias the scalar oracle, so this still
// holds trivially and exercises the dispatch under the per-arch qemu jobs.
func TestBinSIMD(t *testing.T) {
	t.Logf("HaveReduceSIMD = %v", HaveReduceSIMD)
	ops := []struct {
		name   string
		simd   func(dst, a, b []float64)
		scalar func(dst, a, b []float64)
	}{
		{"add", addBin, Add},
		{"sub", subBin, Sub},
		{"mul", mulBin, Mul},
		{"div", divBin, Div},
	}
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 100, 1000, 4096, 9999} {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i] = r.NormFloat64() * 1e3
			b[i] = r.NormFloat64() * 1e3
			if b[i] == 0 {
				b[i] = 1 // keep div well-defined for the random body
			}
		}
		for _, op := range ops {
			got := make([]float64, n)
			want := make([]float64, n)
			op.simd(got, a, b)
			op.scalar(want, a, b)
			for i := range want {
				if !bitEq(got[i], want[i]) {
					t.Fatalf("%s n=%d [%d]: simd=%v, scalar=%v (a=%v b=%v)",
						op.name, n, i, got[i], want[i], a[i], b[i])
				}
			}
		}
	}
	// Edge inputs: signed zeros, infinities, NaN, extremes — paired so each op
	// hits e.g. Inf-Inf=NaN, 0*Inf=NaN, x/0=Inf, 0/0=NaN, -0 handling, etc.
	ea := []float64{-0.0, 0.0, 1, -1, math.Inf(1), math.Inf(-1), math.NaN(), 1e308, -1e308, math.MaxFloat64, math.SmallestNonzeroFloat64, 0.0}
	eb := []float64{0.0, -0.0, math.Inf(1), math.Inf(-1), math.Inf(1), 1, 2, 1e308, 1e308, 2, 2, 0.0}
	for _, op := range ops {
		got := make([]float64, len(ea))
		want := make([]float64, len(ea))
		op.simd(got, ea, eb)
		op.scalar(want, ea, eb)
		for i := range want {
			if !bitEq(got[i], want[i]) {
				t.Fatalf("%s edge[%d] a=%v b=%v: simd=%v, scalar=%v",
					op.name, i, ea[i], eb[i], got[i], want[i])
			}
		}
	}
}

// bitEq reports whether two float64 are bit-identical, treating all NaNs as
// equal to each other (we only require "is a NaN", not a specific payload).
func bitEq(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// TestSqrtSIMD asserts the SIMD sqrt kernel is BIT-IDENTICAL to the scalar
// math.Sqrt oracle across many lengths (vector body + scalar tail of every
// residue mod 8) and across the IEEE edge cases — negatives (-> NaN), +Inf,
// signed zeros, and an explicit NaN input. Unlike the sum reduction, sqrt is a
// single correctly-rounded operation with no grouping freedom, so bit-identity
// (not just closeness) is the contract and SQRTPD must match math.Sqrt exactly.
func TestSqrtSIMD(t *testing.T) {
	t.Logf("HaveReduceSIMD = %v", HaveReduceSIMD)
	r := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 100, 1000, 4096, 9999} {
		src := make([]float64, n)
		for i := range src {
			src[i] = r.Float64() * 1e6
		}
		got := make([]float64, n)
		want := make([]float64, n)
		sqrtSIMD(got, src)
		sqrtScalar(want, src)
		for i := range want {
			if !bitEq(got[i], want[i]) {
				t.Fatalf("n=%d [%d]: sqrtSIMD=%v, scalar=%v", n, i, got[i], want[i])
			}
		}
	}
	// Edge cases: negatives, zeros, infinities, NaN must match the oracle bitwise.
	edge := []float64{
		-1, -0.0, 0.0, 1, 4, 2, 1e308,
		math.Inf(1), math.Inf(-1), math.NaN(), -math.MaxFloat64, math.SmallestNonzeroFloat64,
	}
	got := make([]float64, len(edge))
	want := make([]float64, len(edge))
	sqrtSIMD(got, edge)
	sqrtScalar(want, edge)
	for i := range want {
		if !bitEq(got[i], want[i]) {
			t.Fatalf("edge[%d]=%v: sqrtSIMD=%v, scalar=%v", i, edge[i], got[i], want[i])
		}
	}
}

// TestMaxMinSIMD asserts the SIMD max/min reductions are BIT-IDENTICAL to the
// scalar Max/Min oracle (which is NaN-propagating, matching numpy.max/min and
// Go's math.Max/Min) across many lengths and across NaN placements: a NaN at the
// start, middle, end, or as the sole element must make BOTH the kernel and the
// oracle return NaN. Without any NaN the result is the exact extreme.
func TestMaxMinSIMD(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 2, 3, 7, 8, 9, 15, 16, 31, 64, 100, 1000, 4096, 9999} {
		a := make([]float64, n)
		for i := range a {
			a[i] = r.NormFloat64() * 1e3
		}
		if g, w := maxSIMD(a), Max(a); !bitEq(g, w) {
			t.Fatalf("n=%d: maxSIMD=%v, Max=%v", n, g, w)
		}
		if g, w := minSIMD(a), Min(a); !bitEq(g, w) {
			t.Fatalf("n=%d: minSIMD=%v, Min=%v", n, g, w)
		}
		// Inject a NaN at several positions; both must propagate it.
		for _, pos := range []int{0, n / 2, n - 1} {
			b := append([]float64(nil), a...)
			b[pos] = math.NaN()
			if g := maxSIMD(b); !math.IsNaN(g) {
				t.Fatalf("n=%d NaN@%d: maxSIMD=%v, want NaN", n, pos, g)
			}
			if g := minSIMD(b); !math.IsNaN(g) {
				t.Fatalf("n=%d NaN@%d: minSIMD=%v, want NaN", n, pos, g)
			}
			// And the oracle agrees it is NaN (semantics match).
			if !math.IsNaN(Max(b)) || !math.IsNaN(Min(b)) {
				t.Fatalf("n=%d NaN@%d: oracle did not propagate NaN", n, pos)
			}
		}
	}
	// Infinities and signed zeros bit-match the oracle.
	special := []float64{math.Inf(-1), -0.0, 0.0, math.Inf(1), 5, -5}
	if g, w := maxSIMD(special), Max(special); !bitEq(g, w) {
		t.Fatalf("special: maxSIMD=%v, Max=%v", g, w)
	}
	if g, w := minSIMD(special), Min(special); !bitEq(g, w) {
		t.Fatalf("special: minSIMD=%v, Min=%v", g, w)
	}
	// Single NaN element.
	if g := maxSIMD([]float64{math.NaN()}); !math.IsNaN(g) {
		t.Fatalf("[NaN]: maxSIMD=%v, want NaN", g)
	}
	if g := minSIMD([]float64{math.NaN()}); !math.IsNaN(g) {
		t.Fatalf("[NaN]: minSIMD=%v, want NaN", g)
	}
}

func BenchmarkSqrtSIMD(b *testing.B) {
	src := make([]float64, 1<<20)
	for i := range src {
		src[i] = float64(i%97) * 0.5
	}
	dst := make([]float64, len(src))
	b.Run("simd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sqrtSIMD(dst, src)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sqrtScalar(dst, src)
		}
	})
}

func BenchmarkMaxSIMD(b *testing.B) {
	a := make([]float64, 1<<20)
	for i := range a {
		a[i] = float64(i%97) * 0.5
	}
	b.Run("simd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = maxSIMD(a)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Max(a)
		}
	})
}

func BenchmarkSumSIMD(b *testing.B) {
	a := make([]float64, 1<<20)
	for i := range a {
		a[i] = float64(i%97) * 0.5
	}
	b.Run("simd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = sumSIMD(a)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = Sum(a)
		}
	})
}
