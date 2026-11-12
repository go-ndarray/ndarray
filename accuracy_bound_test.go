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

// sumDepth is the longest chain of roundings in NumPy's pairwise sum
// (numpy/_core/src/umath/loops_utils.h.src, pairwise_sum), which Sum follows,
// computed on the same tree rather than by a closed form:
//   - n < 8: a serial loop, n−1 additions;
//   - n ≤ 128: eight accumulators over the first n − n%8 elements (m/8 − 1
//     additions each), three levels combining them, then the n%8 remaining
//     elements added one by one: up to 14 + 3 + 7 = 24 at n = 127;
//   - otherwise: split at n2 = n/2 rounded down to a multiple of 8, one more
//     addition joining the halves.
//
// Any other order whose tree is at most this deep is within the same bound.
func sumDepth(n int) int {
	switch {
	case n < 8:
		return max(n-1, 0)
	case n <= 128:
		m := n - n%8
		return (m/8 - 1) + 3 + n%8
	default:
		n2 := n / 2
		n2 -= n2 % 8
		return 1 + max(sumDepth(n2), sumDepth(n-n2))
	}
}

// within reports whether |got − want| ≤ bound, with want the exact value: the
// difference is taken in big.Float, so the oracle is never rounded first.
func within(got float64, want *big.Float, bound float64) (bool, float64) {
	d := new(big.Float).SetPrec(oraclePrec).Sub(big.NewFloat(got), want)
	e, _ := d.Abs(d).Float64()
	return e <= bound, e
}

func exactSum(xs []float64) (sum *big.Float, abs float64) {
	s, a := new(big.Float).SetPrec(oraclePrec), new(big.Float).SetPrec(oraclePrec)
	for _, x := range xs {
		s.Add(s, big.NewFloat(x))
		a.Add(a, big.NewFloat(math.Abs(x)))
	}
	abs, _ = a.Float64()
	return s, abs
}

func TestSumDepth(t *testing.T) {
	for _, c := range []struct{ n, d int }{
		{1, 0}, {7, 6}, {8, 3}, {127, 24}, {128, 18}, {129, 1 + 11}, {247, 1 + 24}, // 129 = 64 + 65; 247 = 120 + 127
	} {
		if got := sumDepth(c.n); got != c.d {
			t.Errorf("sumDepth(%d) = %d, want %d", c.n, got, c.d)
		}
	}
}

// TestAccuracyBoundSum: |Sum(x) − Σx| ≤ γ_d·Σ|x|, d = sumDepth(n), in float64
// and in float32 (whose sum accumulates in float32, as NumPy's does). The
// sizes include leaves of 121–127 elements (n = 127, 247), the deepest ones.
func TestAccuracyBoundSum(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 7, 127, 128, 129, 247, 1000, 65536, 1 << 20} {
		xs := make([]float64, n)
		x32 := make([]float32, n)
		for i := range xs {
			// Mixed signs and magnitudes in 2^±4, so cancellation is
			// exercised and the bound stays far below a typical element.
			v := (r.Float64()*2 - 1) * math.Ldexp(1, r.IntN(8)-4)
			x32[i] = float32(v)
			xs[i] = v
		}
		a, err := ndarray.FromData(xs, n)
		if err != nil {
			t.Fatal(err)
		}
		want, abs := exactSum(xs)
		if ok, e := within(a.Sum(), want, gamma(sumDepth(n), 0x1p-53)*abs); !ok {
			t.Errorf("float64 n=%d: |Sum − exact| = %g > bound %g", n, e, gamma(sumDepth(n), 0x1p-53)*abs)
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
		// float32 → float64 is exact, so the depth is the same.
		if ok, e := within(a32.Sum(), want, gamma(sumDepth(n), 0x1p-24)*abs); !ok {
			t.Errorf("float32 n=%d: |Sum − exact| = %g > bound %g", n, e, gamma(sumDepth(n), 0x1p-24)*abs)
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
					abs, _ := ab.Float64()
					if ok, e := within(got[i*n+j], c, gamma(k+extra, u)*abs); !ok {
						bound := gamma(k+extra, u) * abs
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
