package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// intMat returns an m*k matrix of small integer-valued doubles (exactly
// representable, and their sums of products are exact within float64 for these
// sizes) so a kernel must match its oracle bit-for-bit.
func intMat(m, k int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	a := make([]float64, m*k)
	for i := range a {
		a[i] = float64(r.Intn(17) - 8)
	}
	return a
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

// TestMatVecP checks mat·vec over both the serial (below matVecThreshold) and
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

// TestMatVecStridedP checks mat·vec on a sub-block of a wider matrix: only
// the first k of every lda elements are read.
func TestMatVecStridedP(t *testing.T) {
	m, k, lda := 5, 6, 11
	a, v := randVec(m*lda, 5), randVec(k, 6)
	got := make([]float64, m)
	MatVecStridedP(got, a, lda, v, m, k)
	for i := 0; i < m; i++ {
		want := 0.0
		for p := 0; p < k; p++ {
			want += a[i*lda+p] * v[p]
		}
		if math.Abs(got[i]-want) > 1e-9*(1+math.Abs(want)) {
			t.Fatalf("MatVecStridedP [%d]: %v != %v", i, got[i], want)
		}
	}
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

// TestDotSIMD holds the SIMD inner product (AVX2/FMA, NEON, or dotRange on the
// other targets) to a sequential oracle: exactly on integer data for every
// length 0..70 (main loop, folds and tail), within the regrouping bound
// n*eps*sum|a_i*b_i| on random data, and NaN/Inf propagation.
func TestDotSIMD(t *testing.T) {
	naive := func(a, b []float64) (s, abs float64) {
		for i := range a {
			s += a[i] * b[i]
			abs += math.Abs(a[i] * b[i])
		}
		return s, abs
	}
	for n := 0; n <= 70; n++ {
		a, b := intMat(1, n, int64(n)), intMat(1, n, int64(n+100))
		if want, _ := naive(a, b); dotSIMD(a, b) != want {
			t.Fatalf("n=%d: %v != %v", n, dotSIMD(a, b), want)
		}
	}
	for _, n := range []int{17, 1000, 4099} {
		a, b := randVec(n, int64(n)), randVec(n, int64(2*n))
		want, abs := naive(a, b)
		if got := dotSIMD(a, b); math.Abs(got-want) > float64(n)*0x1p-52*abs {
			t.Errorf("n=%d: %v vs %v", n, got, want)
		}
	}
	for _, special := range []float64{math.NaN(), math.Inf(1)} {
		for _, at := range []int{0, 20, 37} {
			a := intMat(1, 38, 5)
			a[at] = special
			b := intMat(1, 38, 6)
			b[at] = 1
			got := dotSIMD(a, b)
			if math.IsNaN(special) != math.IsNaN(got) || !math.IsNaN(special) && !math.IsInf(got, 1) {
				t.Errorf("%v at %d: dot = %v", special, at, got)
			}
		}
	}
}
