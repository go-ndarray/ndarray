package ndarray

import (
	"math"
	"testing"
)

func eqClose(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("[%d] = %v, want %v (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
}

func TestMathUfuncs(t *testing.T) {
	// All expectations from numpy 2.2.4.
	a := mustArr(t, ok(FromData([]float64{0, 1, 4, 9}, 4)))
	eqFloats(t, a.Sqrt().materialize(), []float64{0, 1, 2, 3})

	eqClose(t, mustArr(t, ok(FromData([]float64{0, 1}, 2))).Exp().materialize(),
		[]float64{1, math.E})
	eqClose(t, mustArr(t, ok(FromData([]float64{1, math.E}, 2))).Log().materialize(),
		[]float64{0, 1})
	eqClose(t, mustArr(t, ok(FromData([]float64{1, 2, 8}, 3))).Log2().materialize(),
		[]float64{0, 1, 3})
	eqClose(t, mustArr(t, ok(FromData([]float64{1, 10, 100}, 3))).Log10().materialize(),
		[]float64{0, 1, 2})

	eqClose(t, mustArr(t, ok(FromData([]float64{0, math.Pi / 2}, 2))).Sin().materialize(),
		[]float64{0, 1})
	eqClose(t, mustArr(t, ok(FromData([]float64{0, math.Pi}, 2))).Cos().materialize(),
		[]float64{1, -1})
	eqClose(t, mustArr(t, ok(FromData([]float64{0}, 1))).Tan().materialize(),
		[]float64{0})

	fc := mustArr(t, ok(FromData([]float64{1.7, -1.2}, 2)))
	eqFloats(t, fc.Floor().materialize(), []float64{1, -2})
	cc := mustArr(t, ok(FromData([]float64{1.2, -1.7}, 2)))
	eqFloats(t, cc.Ceil().materialize(), []float64{2, -1})

	// Round here is math.Round (half away from zero); this intentionally differs
	// from numpy's banker's rounding, so 0.5 -> 1 and 2.5 -> 3.
	rc := mustArr(t, ok(FromData([]float64{0.5, 1.5, 2.5, -0.5}, 4)))
	eqFloats(t, rc.Round().materialize(), []float64{1, 2, 3, -1})

	sq := mustArr(t, ok(FromData([]float64{2, -3}, 2)))
	eqFloats(t, sq.Square().materialize(), []float64{4, 9})
	pw := mustArr(t, ok(FromData([]float64{2, 3}, 2)))
	eqFloats(t, pw.Power(3).materialize(), []float64{8, 27})
}

func TestComparisonUfuncs(t *testing.T) {
	x := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	y := mustArr(t, ok(FromData([]float64{3, 2, 1}, 3)))

	cases := []struct {
		name string
		fn   func(*Array) (*Array, error)
		want []float64
	}{
		{"gt", x.Greater, []float64{0, 0, 1}},
		{"ge", x.GreaterEqual, []float64{0, 1, 1}},
		{"lt", x.Less, []float64{1, 0, 0}},
		{"le", x.LessEqual, []float64{1, 1, 0}},
		{"eq", x.Equal, []float64{0, 1, 0}},
		{"ne", x.NotEqual, []float64{1, 0, 1}},
		{"max", x.Maximum, []float64{3, 2, 3}},
		{"min", x.Minimum, []float64{1, 2, 1}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := mustArr(t, ok(c.fn(y)))
			eqFloats(t, r.materialize(), c.want)
		})
	}
}

func TestComparisonBroadcastAndError(t *testing.T) {
	// Comparison against a scalar broadcasts; m > 2 over arange(6).reshape(2,3).
	m := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	two := scalarArray(2)
	gt := mustArr(t, ok(m.Greater(two)))
	eqInts(t, gt.Shape(), []int{2, 3})
	eqFloats(t, gt.materialize(), []float64{0, 0, 0, 1, 1, 1})

	// Mask composes with arithmetic: keep only elements > 2.
	kept := mustArr(t, ok(gt.Mul(m)))
	eqFloats(t, kept.materialize(), []float64{0, 0, 0, 3, 4, 5})

	// Incompatible shapes surface ErrBroadcast through binOp.
	bad := mustArr(t, ok(FromData([]float64{1, 2}, 2)))
	for _, fn := range []func(*Array) (*Array, error){
		m.Greater, m.GreaterEqual, m.Less, m.LessEqual,
		m.Equal, m.NotEqual, m.Maximum, m.Minimum,
	} {
		if _, err := fn(bad); err == nil {
			t.Fatalf("expected broadcast error")
		}
	}
}
