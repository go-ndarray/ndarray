package ndarray

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// TestArangeMatchesNumpy pins Arange to numpy's algorithm: the length is
// ceil((stop-start)/step) and element i is start + i*((start+step)-start),
// with the product rounded before the sum. Expectations are numpy 2.0.2 on
// x86_64; numpy's own arm64 builds fuse that expression into an FMA and can
// differ in the last bit, which is why the Go side forbids fusion. The
// accumulating loop this replaces returned 3 elements for (1, 1.3, 0.1) and 29
// for (0, 2.9000000000000004, 0.1), and never returned for (1e16, 1e16+6, 1).
func TestArangeMatchesNumpy(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range []struct {
		start, stop, step float64
		want              []float64
	}{
		{0x1.0000000000000p+0, 0x1.4cccccccccccdp+0, 0x1.999999999999ap-4, []float64{0x1.0000000000000p+0, 0x1.199999999999ap+0, 0x1.3333333333334p+0, 0x1.4cccccccccccep+0}},
		{0x0.0p+0, 0x1.7333333333334p+1, 0x1.999999999999ap-4, []float64{0x0.0p+0, 0x1.999999999999ap-4, 0x1.999999999999ap-3, 0x1.3333333333334p-2, 0x1.999999999999ap-2, 0x1.0000000000000p-1, 0x1.3333333333334p-1, 0x1.6666666666667p-1, 0x1.999999999999ap-1, 0x1.ccccccccccccdp-1, 0x1.0000000000000p+0, 0x1.199999999999ap+0, 0x1.3333333333334p+0, 0x1.4cccccccccccdp+0, 0x1.6666666666667p+0, 0x1.8000000000000p+0, 0x1.999999999999ap+0, 0x1.b333333333334p+0, 0x1.ccccccccccccdp+0, 0x1.e666666666667p+0, 0x1.0000000000000p+1, 0x1.0cccccccccccdp+1, 0x1.199999999999ap+1, 0x1.2666666666667p+1, 0x1.3333333333334p+1, 0x1.4000000000000p+1, 0x1.4cccccccccccdp+1, 0x1.599999999999ap+1, 0x1.6666666666667p+1, 0x1.7333333333334p+1}},
		{0x1.999999999999ap-4, -0x1.0999999910294p+3, -0x1.3333333333333p-2, []float64{0x1.999999999999ap-4, -0x1.9999999999999p-3, -0x1.0000000000000p-1, -0x1.9999999999999p-1, -0x1.1999999999999p+0, -0x1.6666666666666p+0, -0x1.b333333333332p+0, -0x1.0000000000000p+1, -0x1.2666666666666p+1, -0x1.4ccccccccccccp+1, -0x1.7333333333333p+1, -0x1.9999999999999p+1, -0x1.bffffffffffffp+1, -0x1.e666666666666p+1, -0x1.0666666666667p+2, -0x1.199999999999ap+2, -0x1.2cccccccccccdp+2, -0x1.4000000000000p+2, -0x1.5333333333333p+2, -0x1.6666666666667p+2, -0x1.799999999999ap+2, -0x1.8cccccccccccdp+2, -0x1.a000000000000p+2, -0x1.b333333333333p+2, -0x1.c666666666666p+2, -0x1.d99999999999ap+2, -0x1.ecccccccccccdp+2, -0x1.0000000000000p+3}},
		// start+step rounds back to start, so delta is 0 and numpy repeats it.
		{1e16, 1e16 + 6, 1, []float64{1e16, 1e16, 1e16, 1e16, 1e16, 1e16}},
		// The quotient is 0 but the span is not: one element iff step points
		// towards stop.
		{0, 1, inf, []float64{0}},
		{0, -1, inf, []float64{}},
		{1, 0, 1, []float64{}},
	} {
		a, err := Arange(c.start, c.stop, c.step)
		if err != nil {
			t.Fatalf("Arange(%v, %v, %v): %v", c.start, c.stop, c.step, err)
		}
		got := a.materialize()
		if len(got) != len(c.want) {
			t.Fatalf("Arange(%v, %v, %v) has %d elements, want %d", c.start, c.stop, c.step, len(got), len(c.want))
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("Arange(%v, %v, %v)[%d] = %v, want %v", c.start, c.stop, c.step, i, got[i], c.want[i])
			}
		}
	}
	nan := math.NaN()
	for _, c := range [][3]float64{
		{0, inf, 1}, {-inf, 0, 1}, {0, -inf, -1}, {0, 1e300, 1e-300},
		{0, nan, 1}, {nan, 1, 1}, {0, 1, nan},
	} {
		if _, err := Arange(c[0], c[1], c[2]); !errors.Is(err, ErrShapeMismatch) {
			t.Errorf("Arange(%v) err = %v, want ErrShapeMismatch", c, err)
		}
	}
}

// TestLinspaceRoundsLikeNumpy checks that each sample rounds i*step before
// adding start, as numpy does. Fused into an FMA (which Go may do on arm64,
// ppc64le and s390x) sample 3 below was 9.73, and the result depended on the
// architecture.
func TestLinspaceRoundsLikeNumpy(t *testing.T) {
	a := mustArr(t, ok(Linspace(0.1, 16.150000000000002, 6)))
	want := []float64{0.1, 3.31, 6.52, 9.729999999999999, 12.94, 16.150000000000002}
	for i, w := range want {
		if got := a.At(i); got != w {
			t.Fatalf("Linspace[%d] = %v, want %v (all: %v)", i, got, w, a)
		}
	}
}

// ulpDistance is the number of representable float64 values between x and y.
func ulpDistance(x, y float64) uint64 {
	ix, iy := int64(math.Float64bits(x)), int64(math.Float64bits(y))
	if ix < 0 {
		ix = math.MinInt64 - ix
	}
	if iy < 0 {
		iy = math.MinInt64 - iy
	}
	if ix < iy {
		ix, iy = iy, ix
	}
	return uint64(ix - iy)
}

// TestLog2Accuracy checks Log2 against numpy (identical on arm64 and x86_64)
// near 1, where math.Log2 loses precision to cancellation: it computes
// log(frac)/ln2 + exp with frac in [0.5, 1), so log2(1.02) is -0.97+1. Over
// 900k random inputs math.Log2 was 4 or more ULP off for 12% of them and up to
// 2.8e7 ULP off just above 1; the reduced form stays within 2 ULP.
func TestLog2Accuracy(t *testing.T) {
	for _, c := range [][2]float64{
		{0x1.0000000400000p+0, 0x1.7154764fd586fp-30},
		{0x1.051eb851eb852p+0, 0x1.d413b59703520p-6},
		{0x1.ff7ced916872bp-1, -0x1.7a6210ffb7151p-10},
		{0x1.000000006df38p+0, 0x1.3d40846f44141p-33},
		{0x1.8000000000000p+0, 0x1.2b803473f7ad1p-1},
		{0x1.8000000000000p-1, -0x1.a8ff971810a5ep-2},
		{0x1.56e1fc2f8f359p-997, -0x1.f24a09f1a8b89p+9},
		{0x1.999999999999ap-4, -0x1.a934f0979a371p+1},
	} {
		x := mustArr(t, ok(FromData([]float64{c[0]}, 1)))
		if got := x.Log2().At(0); ulpDistance(got, c[1]) > 2 {
			t.Errorf("Log2(%v) = %v, want %v (%d ULP)", c[0], got, c[1], ulpDistance(got, c[1]))
		}
	}
	// Exact and special values.
	nan, inf := math.NaN(), math.Inf(1)
	x := mustArr(t, ok(FromData([]float64{1, 2, 8, 0.25, 0x1p-1074, 0, math.Copysign(0, -1), inf, -1, nan}, 10)))
	want := []float64{0, 1, 3, -2, -1074, -inf, -inf, inf, nan, nan}
	got := x.Log2().materialize()
	for i := range want {
		if got[i] != want[i] && !(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
			t.Errorf("Log2(%v) = %v, want %v", x.At(i), got[i], want[i])
		}
	}
	// Complex: log(z)/ln 2, as numpy.log2 (log2(-1) = iπ/ln 2).
	z := data[complex128](t, mustArr(t, ok(FromSlice([]complex128{8, -1}, 2))).Log2())
	if cmplx.Abs(z[0]-3) > 1e-15 || cmplx.Abs(z[1]-complex(0, math.Pi/math.Ln2)) > 1e-15 {
		t.Errorf("complex Log2 = %v", z)
	}
}

// TestFloat32ExtremaMatchTheFloat64Path: the float32 Max/Min read the float32
// elements directly; converting to float64 first (the path they replace) gives
// the same value, since the conversion is exact. Strided views are included.
func TestFloat32ExtremaMatchTheFloat64Path(t *testing.T) {
	nan := float32(math.NaN())
	for _, d := range [][]float32{
		{3}, {1, -2, 5.5, 0.25}, {-1, -7, float32(math.Copysign(0, -1)), -3, 0},
		{1, nan, 3}, {float32(math.Inf(-1)), -1e30, 1e-45},
	} {
		a := mustArr(t, ok(FromSlice(d, len(d))))
		for _, v := range []*Array{a, mustArr(t, ok(a.Slice(Step(-1))))} {
			mx, _ := v.Max()
			mn, _ := v.Min()
			wmx, _ := v.AsType(Float64).Max()
			wmn, _ := v.AsType(Float64).Min()
			if !sameFloat(mx, wmx) || !sameFloat(mn, wmn) {
				t.Errorf("%v: Max=%v Min=%v, float64 path %v %v", d, mx, mn, wmx, wmn)
			}
		}
	}
	if _, err := mustArr(t, ok(ZerosOf(Float32, 0))).Max(); err == nil {
		t.Error("Max of an empty float32 array: no error")
	}
	if _, err := mustArr(t, ok(ZerosOf(Float32, 0))).Min(); err == nil {
		t.Error("Min of an empty float32 array: no error")
	}
}

// sameFloat is bit equality, any NaN matching any NaN.
func sameFloat(a, b float64) bool {
	if a != a && b != b {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// TestExtremaOfSignedZeros: Max is +0 and Min is -0 over a mix of signed
// zeros, whatever their order, as Go's max/min and math.Max/Min order them.
// Until v0.19.1 amd64 returned the sign of whichever zero its kernel met last
// (MAXPD/MINPD return their source operand on a zero tie). The tests had meant
// to cover this but wrote -0.0, which Go's constant arithmetic makes +0.
func TestExtremaOfSignedZeros(t *testing.T) {
	nz := math.Copysign(0, -1)
	for _, n := range []int{2, 3, 9, 16, 17, 40} {
		for first := 0; first < 2; first++ {
			d := make([]float64, n)
			for i := range d {
				if (i+first)%2 == 0 {
					d[i] = nz
				}
			}
			a := mustArr(t, ok(FromData(d, n)))
			mx, _ := a.Max()
			mn, _ := a.Min()
			if math.Signbit(mx) || !math.Signbit(mn) {
				t.Errorf("n=%d first=%d: Max=%v Min=%v, want 0 and -0", n, first, mx, mn)
			}
			rows := mustArr(t, ok(a.Reshape(1, n)))
			rmx := mustArr(t, ok(rows.MaxAxis(1, false))).materialize()[0]
			rmn := mustArr(t, ok(rows.MinAxis(1, false))).materialize()[0]
			if math.Signbit(rmx) || !math.Signbit(rmn) {
				t.Errorf("n=%d first=%d: MaxAxis=%v MinAxis=%v, want 0 and -0", n, first, rmx, rmn)
			}
		}
	}
}
