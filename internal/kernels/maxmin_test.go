package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// maxminPool is the value table: zeros of both signs, infinities, subnormals,
// neighbours of 1, and random bit patterns that are not NaN.
func maxminPool() []float64 {
	v := []float64{
		0, math.Copysign(0, -1), 1, -1, math.Inf(1), math.Inf(-1),
		5e-324, -5e-324, 2.2250738585072009e-308, -2.2250738585072014e-308,
		math.MaxFloat64, -math.MaxFloat64, math.Nextafter(1, 2), math.Nextafter(-1, -2),
		0.5, -0.5, 1e-300, -1e-300,
	}
	rng := rand.New(rand.NewSource(7))
	for len(v) < 600 {
		x := math.Float64frombits(rng.Uint64())
		if !math.IsNaN(x) {
			v = append(v, x)
		}
	}
	return v
}

func maxminCheck(t *testing.T, name string, k func(dst, a, b []float64), ref func(x, y float64) float64) {
	t.Helper()
	pool := maxminPool()
	lengths := make([]int, 0, 36)
	for n := 0; n <= 33; n++ {
		lengths = append(lengths, n)
	}
	lengths = append(lengths, ParThreshold-1, ParThreshold+37, 100000)
	for _, n := range lengths {
		for off := 0; off < 4; off++ {
			a, b := make([]float64, n), make([]float64, n)
			for i := range a {
				a[i] = pool[(off*13+i*7)%len(pool)]
				b[i] = pool[(off*29+i*11+3)%len(pool)]
			}
			if n >= 36 { // every ordered pair of the first 18 values, zeros in both orders
				for i := 0; i < 18*18 && i < n; i++ {
					a[i], b[i] = pool[i/18], pool[i%18]
				}
			}
			check := func(label string, got []float64) {
				for i := range got {
					if w := ref(a[i], b[i]); math.Float64bits(got[i]) != math.Float64bits(w) {
						t.Fatalf("%s %s n=%d [%d] (%#x, %#x): got %#x want %#x", name, label, n, i,
							math.Float64bits(a[i]), math.Float64bits(b[i]),
							math.Float64bits(got[i]), math.Float64bits(w))
					}
				}
			}
			dst := make([]float64, n)
			k(dst, a, b)
			check("fresh", dst)

			// Aliasing: out = a, then out = b (the reference reads the originals).
			a0, b0 := append([]float64(nil), a...), append([]float64(nil), b...)
			k(a, a, b)
			got := append([]float64(nil), a...)
			copy(a, a0)
			check("out=a", got)
			k(b, a, b)
			got = append([]float64(nil), b...)
			copy(b, b0)
			check("out=b", got)
		}
	}
}

// nanCheck: a NaN in either operand gives a NaN. Which NaN is target-dependent.
func nanCheck(t *testing.T, name string, k func(dst, a, b []float64)) {
	t.Helper()
	nan := math.NaN()
	snan := math.Float64frombits(0x7FF0000000000001)
	vals := []float64{0, math.Copysign(0, -1), 1, math.Inf(1), math.Inf(-1), 5e-324}
	for _, n := range []int{1, 2, 3, 4, 7, 8, 9, 16, 17, 33, 100} {
		for _, bad := range []float64{nan, snan, -nan} {
			for pos := 0; pos < n; pos += 3 {
				for _, inA := range []bool{true, false} {
					a, b, dst := make([]float64, n), make([]float64, n), make([]float64, n)
					for i := range a {
						a[i], b[i] = vals[i%len(vals)], vals[(i+2)%len(vals)]
					}
					if inA {
						a[pos] = bad
					} else {
						b[pos] = bad
					}
					k(dst, a, b)
					for i := range dst {
						want := i == pos
						if math.IsNaN(dst[i]) != want {
							t.Fatalf("%s n=%d pos=%d [%d]: IsNaN = %v, want %v", name, n, pos, i, math.IsNaN(dst[i]), want)
						}
					}
				}
			}
		}
		a, b, dst := make([]float64, n), make([]float64, n), make([]float64, n) // both NaN
		for i := range a {
			a[i], b[i] = nan, math.Float64frombits(0x7FF0000000000001)
		}
		k(dst, a, b)
		for i := range dst {
			if !math.IsNaN(dst[i]) {
				t.Fatalf("%s n=%d: NaN, NaN gave %v", name, n, dst[i])
			}
		}
	}
}

func TestMaximumMinimumFloats(t *testing.T) {
	maxminCheck(t, "Maximum", Maximum, func(x, y float64) float64 { return max(x, y) })
	maxminCheck(t, "Minimum", Minimum, func(x, y float64) float64 { return min(x, y) })
	nanCheck(t, "Maximum", Maximum)
	nanCheck(t, "Minimum", Minimum)
}
