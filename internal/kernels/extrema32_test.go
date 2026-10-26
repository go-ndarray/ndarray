package kernels

import (
	"math"
	"testing"
)

// seqMax32 and seqMin32 are the plainest oracle: one sequential pass with Go's
// builtin max and min, which propagate NaN and order -0 below +0.
func seqMax32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		m = max(m, v)
	}
	return m
}

func seqMin32(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		m = min(m, v)
	}
	return m
}

// TestExtrema32MatchSequential: Max32/Min32 (the kernel where there is one)
// and the four-chain Go loops agree with one sequential pass, bit for bit, at
// every length around the blocks, with NaN, signed zeros and infinities placed
// at every position.
func TestExtrema32MatchSequential(t *testing.T) {
	nz := float32(math.Copysign(0, -1))
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	check := func(name string, a []float32) {
		t.Helper()
		for _, f := range []struct {
			name      string
			got, want float32
		}{
			{"Max32", Max32(a), seqMax32(a)},
			{"Min32", Min32(a), seqMin32(a)},
			{"max32Go", max32Go(a), seqMax32(a)},
			{"min32Go", min32Go(a), seqMin32(a)},
		} {
			if !sameBits32(f.got, f.want) {
				t.Fatalf("%s %s n=%d: %v, sequential %v", name, f.name, len(a), f.got, f.want)
			}
		}
	}
	for n := 1; n <= 70; n++ {
		check("mixed", operands32(n, 5))
		for pos := 0; pos < n; pos++ {
			for _, fill := range []float32{nz, 0, -1, 1} {
				for _, v := range []float32{nz, 0, inf, -inf, nan} {
					a := make([]float32, n)
					for i := range a {
						a[i] = fill
					}
					a[pos] = v
					check("special", a)
				}
			}
		}
	}
}

func TestExtrema32SignedZeros(t *testing.T) {
	nz := float32(math.Copysign(0, -1))
	for n := 2; n <= 40; n++ {
		for first := 0; first < 2; first++ {
			a := make([]float32, n)
			for i := range a {
				if (i+first)%2 == 0 {
					a[i] = nz
				}
			}
			if mx, mn := Max32(a), Min32(a); math.Signbit(float64(mx)) || !math.Signbit(float64(mn)) {
				t.Fatalf("n=%d first=%d: Max32=%v Min32=%v, want 0 and -0", n, first, mx, mn)
			}
		}
	}
}
