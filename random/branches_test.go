package random

import (
	"math"
	"testing"
)

// stubBG replays a fixed list of 64-bit outputs (cycling), so that tests can
// steer a sampler into branches that real streams reach once in billions of
// draws. Uint32 is the low half of the next output, Float64 its top 53 bits.
type stubBG struct {
	u []uint64
	i *int
}

func newStub(u ...uint64) stubBG { return stubBG{u, new(int)} }

func (s stubBG) Uint64() uint64 {
	v := s.u[*s.i%len(s.u)]
	*s.i++
	return v
}
func (s stubBG) Uint32() uint32   { return uint32(s.Uint64()) }
func (s stubBG) Float64() float64 { return uint64ToDouble(s.Uint64()) }
func (s stubBG) Raw() uint64      { return s.Uint64() }

// f64 returns the output whose Float64 is u (for u a multiple of 2^-53).
func f64(u float64) uint64 { return uint64(u*(1<<53)) << 11 }

func TestRareBranches(t *testing.T) {
	// Lemire rejections: a low product below the threshold redraws.
	if v := lemire64(newStub(0, 1), 3<<62-1); v != 0 {
		t.Errorf("lemire64 = %d", v)
	}
	if v := lemire32(newStub(0, 1), 3<<30-1); v != 0 {
		t.Errorf("lemire32 = %d", v)
	}
	if v := boundedUint64(newStub(0, 1), 7, 3<<62-1, false); v != 7 {
		t.Errorf("boundedUint64 = %d", v)
	}
	// 64-bit masked interval redraws above max.
	if v := interval(newStub(^uint64(0), 5), 1<<40); v != 5 {
		t.Errorf("interval = %d", v)
	}
	out := make([]uint32, 3)
	fillUint32(newStub(15, 3), 7, 0, true, out[:1])
	fillUint32(newStub(15, 3), 0, 9, true, out[1:])
	if out[0] != 7 || out[1] != 3 {
		t.Errorf("fillUint32 = %v", out)
	}

	// Beta (Johnk) when X or Y underflows, in both orders.
	tiny := f64(0x1p-40)
	if v := beta(newStub(tiny, f64(0.5)), 1e-3, 1e-3); !(v >= 0 && v <= 1) {
		t.Errorf("beta = %v", v)
	}
	if v := beta(newStub(f64(0.5), tiny), 1e-3, 1e-3); !(v >= 0 && v <= 1) {
		t.Errorf("beta = %v", v)
	}

	// BTPE: the region-3 and region-4 rejections (y < 0, y > n).
	var bc binomialCache
	binomialBTPE(newStub(0), 1000, 0.4, &bc, false)
	u3 := (bc.p2 + bc.p3) / 2 / bc.p4
	u4 := (bc.p3 + bc.p4) / 2 / bc.p4
	acc := f64(0)
	if y := binomialBTPE(newStub(f64(math.Floor(u3*(1<<53))/(1<<53)), 0, acc, acc), 1000, 0.4, &bc, false); y < 0 || y > 1000 {
		t.Errorf("btpe = %d", y)
	}
	if y := binomialBTPE(newStub(f64(math.Floor(u4*(1<<53))/(1<<53)), 0, acc, acc), 1000, 0.4, &bc, false); y < 0 || y > 1000 {
		t.Errorf("btpe = %d", y)
	}
	// Inversion past its bound starts over.
	bc = binomialCache{}
	if x := binomialInversion(newStub(f64(1-0x1p-53), f64(0)), 10, 0.1, &bc, false); x != 0 {
		t.Errorf("inversion = %d", x)
	}

	// NaN parameters.
	if !math.IsNaN(noncentralChisquare(newStub(1), 2, math.NaN())) || !math.IsNaN(vonmises(newStub(1), 0, math.NaN())) {
		t.Error("NaN parameter")
	}
	// Logseries: V == 0 is rejected.
	if v := logseries(newStub(f64(0), f64(0.5), f64(0.9)), 0.5); v != 1 {
		t.Errorf("logseries = %d", v)
	}
	// Zipf's a >= 1025 shortcut.
	if zipf(newStub(1), 2000) != 1 {
		t.Error("zipf")
	}
	// Geometric beyond int64.
	if geometricInversion(NewPCG64(1), 1e-300) != math.MaxInt64 {
		t.Error("geometric overflow")
	}
	// Laplace, Gumbel and Logistic reject an endpoint draw.
	for _, f := range []func(BitGenerator) float64{
		func(b BitGenerator) float64 { return laplace(b, 0, 1) },
		func(b BitGenerator) float64 { return gumbel(b, 0, 1) },
		func(b BitGenerator) float64 { return logistic(b, 0, 1) },
	} {
		if v := f(newStub(0, f64(0.5))); math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("endpoint draw gave %v", v)
		}
	}
}

func TestLibmSpecialValues(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"log(0)", log(0), math.Inf(-1)},
		{"log(inf)", log(inf), inf},
		{"log1p(-1)", log1p(-1), math.Inf(-1)},
		{"log1p(inf)", log1p(inf), inf},
		{"exp(800)", exp(800), inf},
		{"exp(-800)", exp(-800), 0},
		{"expm1(800)", expm1(800), inf},
		{"expm1(-50)", expm1(-50), -1},
		{"expm1(705)", expm1(705), exp(705)},
		{"pow(10, 400)", pow(10, 400), inf},
		{"pow(10, -400)", pow(10, -400), 0},
		{"pow(0, 2)", pow(0, 2), 0},
		{"log1p(0)", log1p(0), 0},
		{"expm1(0)", expm1(0), 0},
	} {
		if c.got != c.want {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	for _, v := range []float64{log(-1), log1p(-2), exp(math.NaN()), expm1(math.NaN()), log(math.NaN())} {
		if !math.IsNaN(v) {
			t.Errorf("want NaN, got %v", v)
		}
	}
	if logf(1) != 0 || expf(0) != 1 || log1pf(0) != 0 || powf(2, 3) != 8 {
		t.Error("float32 wrappers")
	}
}
