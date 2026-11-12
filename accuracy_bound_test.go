package ndarray_test

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The accuracy contract (README, "Accuracy contract") asserted against an exact
// oracle: math/big at 1024 bits, wide enough that every sum and dot product of
// these inputs is exact before its final rounding. The bounds are the rigorous
// worst-case bounds of the algorithm (Higham, "Accuracy and Stability of
// Numerical Algorithms", 2nd ed., §3.1 and §4.2), so they hold for any
// summation order and with or without FMA: a faster default that reorders or
// fuses is accepted as long as these tests pass.

const oraclePrec = 1024

// gamma is Higham's γ_m = m·u/(1 − m·u), with u the unit roundoff.
func gamma(m int, u float64) float64 { return float64(m) * u / (1 - float64(m)*u) }

// sumDepth is the longest chain of roundings in NumPy's pairwise sum, which
// Sum follows: blocks of at most 128 elements summed by eight accumulators
// (up to 16 serial additions each, then three levels combining them), and the
// blocks combined pairwise. Any reordering with a depth at most this one is
// within the same bound.
func sumDepth(n int) int {
	d := 16 + 3
	for blocks := (n + 127) / 128; blocks > 1; blocks = (blocks + 1) / 2 {
		d++
	}
	return d
}

func exactSum(xs []float64) (sum, abs float64) {
	s, a := new(big.Float).SetPrec(oraclePrec), new(big.Float).SetPrec(oraclePrec)
	for _, x := range xs {
		s.Add(s, big.NewFloat(x))
		a.Add(a, big.NewFloat(math.Abs(x)))
	}
	sum, _ = s.Float64()
	abs, _ = a.Float64()
	return sum, abs
}

// TestAccuracyBoundSum: |Sum(x) − Σx| ≤ γ_d·Σ|x|, d = sumDepth(n), in float64
// and in float32 (whose sum accumulates in float32, as NumPy's does).
func TestAccuracyBoundSum(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 7, 128, 129, 1000, 65536, 1 << 20} {
		xs := make([]float64, n)
		x32 := make([]float32, n)
		for i := range xs {
			// Mixed signs and magnitudes, so cancellation is exercised.
			// The range is kept to 2^±4 so that the bound stays below the
			// smallest element: a dropped or doubled element fails it.
			v := (r.Float64()*2 - 1) * math.Ldexp(1, r.IntN(8)-4)
			if v == 0 {
				v = 1
			}
			x32[i] = float32(v)
			xs[i] = v
		}
		a, err := ndarray.FromData(xs, n)
		if err != nil {
			t.Fatal(err)
		}
		want, abs := exactSum(xs)
		if got, bound := a.Sum(), gamma(sumDepth(n), 0x1p-53)*abs; math.Abs(got-want) > bound {
			t.Errorf("float64 n=%d: |Sum − exact| = %g > bound %g", n, math.Abs(got-want), bound)
		}

		a32, err := ndarray.FromSlice(x32, n)
		if err != nil {
			t.Fatal(err)
		}
		xs32 := make([]float64, n)
		for i, v := range x32 {
			xs32[i] = float64(v)
		}
		want, abs = exactSum(xs32)
		// One more rounding: the float32 result is returned as a float64.
		if got, bound := a32.Sum(), gamma(sumDepth(n)+1, 0x1p-24)*abs; math.Abs(got-want) > bound {
			t.Errorf("float32 n=%d: |Sum − exact| = %g > bound %g", n, math.Abs(got-want), bound)
		}
	}
}

// TestAccuracyBoundMatMul: every element of C = A·B satisfies
// |Ĉij − Cij| ≤ γ_k·(|A|·|B|)ij, the bound of any inner-product order (blocked,
// packed, SIMD, with or without FMA), in float64 and float32.
func TestAccuracyBoundMatMul(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, s := range [][3]int{{1, 1, 1}, {7, 13, 5}, {64, 64, 64}, {97, 300, 33}, {256, 256, 256}} {
		m, k, n := s[0], s[1], s[2]
		av, bv := make([]float64, m*k), make([]float64, k*n)
		a32, b32 := make([]float32, m*k), make([]float32, k*n)
		for i := range av {
			a32[i] = float32(r.NormFloat64())
			av[i] = float64(a32[i])
		}
		for i := range bv {
			b32[i] = float32(r.NormFloat64())
			bv[i] = float64(b32[i])
		}
		check := func(name string, got []float64, u float64, extra int) {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					c, ab := new(big.Float).SetPrec(oraclePrec), new(big.Float).SetPrec(oraclePrec)
					p := new(big.Float).SetPrec(oraclePrec)
					for q := 0; q < k; q++ {
						p.Mul(big.NewFloat(av[i*k+q]), big.NewFloat(bv[q*n+j]))
						c.Add(c, p)
						ab.Add(ab, p.Abs(p))
					}
					want, _ := c.Float64()
					abs, _ := ab.Float64()
					if e, bound := math.Abs(got[i*n+j]-want), gamma(k+extra, u)*abs; e > bound {
						t.Fatalf("%s %dx%dx%d C[%d,%d]: error %g > bound %g", name, m, k, n, i, j, e, bound)
					}
				}
			}
		}

		A, _ := ndarray.FromData(av, m, k)
		B, _ := ndarray.FromData(bv, k, n)
		C, err := A.MatMul(B)
		if err != nil {
			t.Fatal(err)
		}
		c64, _ := ndarray.Data[float64](C)
		check("float64", c64, 0x1p-53, 0)

		A32, _ := ndarray.FromSlice(a32, m, k)
		B32, _ := ndarray.FromSlice(b32, k, n)
		C32, err := A32.MatMul(B32)
		if err != nil {
			t.Fatal(err)
		}
		d32, ok := ndarray.Data[float32](C32)
		if !ok {
			t.Fatalf("float32 MatMul returned %v, not float32", C32.DType())
		}
		c32 := make([]float64, len(d32))
		for i, v := range d32 {
			c32[i] = float64(v)
		}
		// A path that computes in float64 and rounds once to float32 is
		// within γ_{k+1} of float32's unit roundoff too.
		check("float32", c32, 0x1p-24, 1)
	}
}
