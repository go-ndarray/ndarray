package kernels

import (
	"math"
	"testing"
)

// TestSumMatchesNumPy holds Sum's value (SumP, sumSIMD) to np.sum bit for bit.
// The golden values are NumPy 2.5.4's (np.sum of a[i] = -1 + 2*frac(0.618033988749
// 8949*(i+1)), computed with np.arange, not by this package):
//
//	uv run --no-project --python 3.12.12 --with numpy==2.5.4 python -I -c \
//	  'import numpy as np; i=np.arange(1000)+1.0; print(float(np.sum(-1+2*((i*0.6180339887498949)%1.0))).hex())'
//
// Since K1 of the go-scikit performance campaign the sum is floats.SumPairwise,
// NumPy's order; the lane-parallel sums it replaced differed in the last bits.
func TestSumMatchesNumPy(t *testing.T) {
	for _, c := range []struct {
		n    int
		want float64
	}{
		{1000, 0x1.748cc9f0cbc00p-6},
		{100000, 0x1.cbaf96d9ef5e0p-1},
	} {
		a := make([]float64, c.n)
		for i := range a {
			ph := float64(float64(i+1) * 0.6180339887498949)
			a[i] = -1 + float64(2*(ph-math.Floor(ph)))
		}
		if got := SumP(a); math.Float64bits(got) != math.Float64bits(c.want) {
			t.Errorf("SumP n=%d: %v (%#x), NumPy %v (%#x)", c.n, got, math.Float64bits(got), c.want, math.Float64bits(c.want))
		}
		if got := sumSIMD(a); math.Float64bits(got) != math.Float64bits(c.want) {
			t.Errorf("sumSIMD n=%d: %v, NumPy %v", c.n, got, c.want)
		}
	}
}

// TestSumPParallelIsBitIdentical: SumP cuts NumPy's pairwise tree and sums the
// subtrees on several cores; the result is floats.SumPairwise's bits for every
// length and core count (a subtree of only -0 included: the final +0 erases it).
func TestSumPParallelIsBitIdentical(t *testing.T) {
	negZero := math.Copysign(0, -1)
	for _, procs := range []int{1, 2, 3, 4, 7, 16} {
		withMaxProcs(procs, func() {
			withThresholds(64, 1<<14, func() {
				for _, n := range []int{64, 129, 1000, 4097, 100000, 1<<20 + 13, 3<<20 - 7} {
					a := randVec(n, int64(n))
					for i := range a {
						a[i] *= 1e8 // wide dynamic range: any regrouping would show
						if i%97 == 0 {
							a[i] = 1e-9
						}
					}
					want := sumSIMD(a)
					if got := SumP(a); math.Float64bits(got) != math.Float64bits(want) {
						t.Fatalf("procs=%d n=%d: SumP %v (%#x), SumPairwise %v (%#x)", procs, n, got, math.Float64bits(got), want, math.Float64bits(want))
					}
					for i := range a {
						a[i] = negZero
					}
					if got := SumP(a); math.Float64bits(got) != math.Float64bits(sumSIMD(a)) {
						t.Fatalf("procs=%d n=%d all -0: SumP %v", procs, n, got)
					}
				}
			})
		})
	}
}
