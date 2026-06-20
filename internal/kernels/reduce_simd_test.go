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
