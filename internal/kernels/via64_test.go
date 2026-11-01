package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// ulps32 is the distance between two float32s in units in the last place
// (0 for equal values, including two NaNs).
func ulps32(a, b float32) int64 {
	if a == b || (a != a && b != b) {
		return 0
	}
	ia, ib := int64(math.Float32bits(a)), int64(math.Float32bits(b))
	if ia < 0 {
		ia = math.MinInt32 - ia
	}
	if ib < 0 {
		ib = math.MinInt32 - ib
	}
	if d := ia - ib; d < 0 {
		return -d
	} else {
		return d
	}
}

// TestVia64MatchesMath holds Exp32 and Log32 to float32(math.Exp/Log(x)):
// both round a float64 result once, so they may differ by one float32 ULP
// where math's float64 result (0.88 ULP) and ours (0.504) fall on opposite
// sides of a rounding midpoint, and must agree on at least 99.9% of inputs.
// Every length 0..600 crosses the 256-element chunk boundaries.
func TestVia64MatchesMath(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for _, f := range []struct {
		name string
		k    func(dst, src []float32)
		ref  func(float64) float64
		gen  func() float32
	}{
		{"Exp", Exp32, math.Exp, func() float32 { return float32(r.Float64()*200 - 110) }},
		{"Log", Log32, math.Log, func() float32 {
			return float32(math.Exp(r.Float64()*170 - 85))
		}},
	} {
		const n = 100000
		src, dst := make([]float32, n), make([]float32, n)
		for i := range src {
			src[i] = f.gen()
		}
		f.k(dst, src)
		var off int
		for i, x := range src {
			want := float32(f.ref(float64(x)))
			switch d := ulps32(dst[i], want); {
			case d > 1:
				t.Fatalf("%s(%v) = %v, math gives %v (%d ULP)", f.name, x, dst[i], want, d)
			case d == 1:
				off++
			}
		}
		if off > n/1000 {
			t.Errorf("%s: %d of %d differ from math by one ULP", f.name, off, n)
		}
		for m := 0; m <= 600; m++ {
			d := make([]float32, m)
			f.k(d, src[:m])
			for i := range d {
				if d[i] != dst[i] {
					t.Fatalf("%s length %d [%d]: %v != %v", f.name, m, i, d[i], dst[i])
				}
			}
		}
	}
}

// TestVia64Specials checks the edges that come from rounding the float64
// result: overflow to +Inf, underflow through the subnormals to 0, and the
// IEEE special values.
func TestVia64Specials(t *testing.T) {
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	negZero := float32(math.Copysign(0, -1))
	for _, c := range []struct {
		name    string
		k       func(dst, src []float32)
		in, out float32
	}{
		{"Exp", Exp32, 0, 1},
		{"Exp", Exp32, negZero, 1},
		{"Exp", Exp32, 89, inf}, // finite in float64, overflows float32
		{"Exp", Exp32, -inf, 0},
		{"Exp", Exp32, inf, inf},
		{"Exp", Exp32, -104, 0}, // below the smallest float32 subnormal
		{"Exp", Exp32, -100, float32(math.Exp(-100))},
		{"Log", Log32, 1, 0},
		{"Log", Log32, 0, -inf},
		{"Log", Log32, negZero, -inf},
		{"Log", Log32, inf, inf},
		{"Log", Log32, math.SmallestNonzeroFloat32, float32(math.Log(math.SmallestNonzeroFloat32))},
	} {
		d := []float32{0}
		c.k(d, []float32{c.in})
		if d[0] != c.out || math.Signbit(float64(d[0])) != math.Signbit(float64(c.out)) {
			t.Errorf("%s(%v) = %v, want %v", c.name, c.in, d[0], c.out)
		}
	}
	for _, in := range []float32{nan, -1, -inf} {
		d := []float32{0}
		Log32(d, []float32{in})
		if d[0] == d[0] {
			t.Errorf("Log(%v) = %v, want NaN", in, d[0])
		}
	}
	d := []float32{0}
	Exp32(d, []float32{nan})
	if d[0] == d[0] {
		t.Errorf("Exp(NaN) = %v", d[0])
	}
}

// TestVia64InPlaceAndParallel: dst may be src, and the parallel entries give
// the serial bits.
func TestVia64InPlaceAndParallel(t *testing.T) {
	src := make([]float32, 3*ParThreshold+17)
	for i := range src {
		src[i] = float32(i%97) * 0.25
	}
	want := make([]float32, len(src))
	Exp32(want, src)
	got := append([]float32(nil), src...)
	Exp32(got, got)
	par := make([]float32, len(src))
	withMaxProcs(4, func() { Exp32P(par, src) })
	for i := range want {
		if got[i] != want[i] || par[i] != want[i] {
			t.Fatalf("[%d]: in place %v, parallel %v, serial %v", i, got[i], par[i], want[i])
		}
	}
	Log32P(par, src) // the serial path below ParThreshold is covered by the tests above
	for i, x := range src {
		if par[i] != float32(math.Log(float64(x))) && ulps32(par[i], float32(math.Log(float64(x)))) > 1 {
			t.Fatalf("Log32P [%d]: %v", i, par[i])
		}
	}
}
