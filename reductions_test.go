package ndarray

import (
	"errors"
	"reflect"
	"testing"
)

// mat is the shared 2x3 fixture; all expectations below are numpy 2.2.4 values.
func mat(t *testing.T) *Array {
	t.Helper()
	a, err := FromData([]float64{1, 5, 3, 9, 2, 8}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func wantData(t *testing.T, a *Array, shape []int, data []float64) {
	t.Helper()
	if !reflect.DeepEqual(a.Shape(), shape) {
		t.Fatalf("shape = %v, want %v", a.Shape(), shape)
	}
	if got := a.materialize(); !reflect.DeepEqual(got, data) {
		t.Fatalf("data = %v, want %v", got, data)
	}
}

func TestArgMaxMin(t *testing.T) {
	a := mat(t)
	if got, _ := a.ArgMax(); got != 3 {
		t.Errorf("ArgMax = %d, want 3", got)
	}
	if got, _ := a.ArgMin(); got != 0 {
		t.Errorf("ArgMin = %d, want 0", got)
	}

	empty, _ := New(0)
	if _, err := empty.ArgMax(); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("ArgMax empty err = %v", err)
	}
	if _, err := empty.ArgMin(); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("ArgMin empty err = %v", err)
	}

	for _, c := range []struct {
		name     string
		fn       func(int, bool) (*Array, error)
		axis     int
		keepdims bool
		shape    []int
		data     []float64
	}{
		{"argmax0", a.ArgMaxAxis, 0, false, []int{3}, []float64{1, 0, 1}},
		{"argmax1", a.ArgMaxAxis, 1, false, []int{2}, []float64{1, 0}},
		{"argmin0", a.ArgMinAxis, 0, false, []int{3}, []float64{0, 1, 0}},
		{"argmin1", a.ArgMinAxis, -1, false, []int{2}, []float64{0, 1}},
		{"argmax0keep", a.ArgMaxAxis, 0, true, []int{1, 3}, []float64{1, 0, 1}},
	} {
		got, err := c.fn(c.axis, c.keepdims)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		wantData(t, got, c.shape, c.data)
	}

	if _, err := a.ArgMaxAxis(2, false); !errors.Is(err, ErrAxis) {
		t.Errorf("ArgMaxAxis bad axis err = %v", err)
	}
	if _, err := a.ArgMinAxis(5, false); !errors.Is(err, ErrAxis) {
		t.Errorf("ArgMinAxis bad axis err = %v", err)
	}
}

func TestCumulative(t *testing.T) {
	a := mat(t)

	cs0, _ := a.CumSum(0)
	wantData(t, cs0, []int{2, 3}, []float64{1, 5, 3, 10, 7, 11})
	cs1, _ := a.CumSum(1)
	wantData(t, cs1, []int{2, 3}, []float64{1, 6, 9, 9, 11, 19})
	cp1, _ := a.CumProd(-1)
	wantData(t, cp1, []int{2, 3}, []float64{1, 5, 15, 9, 18, 144})

	wantData(t, a.CumSumFlat(), []int{6}, []float64{1, 6, 9, 18, 20, 28})
	wantData(t, a.CumProdFlat(), []int{6}, []float64{1, 5, 15, 135, 270, 2160})

	if _, err := a.CumSum(9); !errors.Is(err, ErrAxis) {
		t.Errorf("CumSum bad axis err = %v", err)
	}
	if _, err := a.CumProd(9); !errors.Is(err, ErrAxis) {
		t.Errorf("CumProd bad axis err = %v", err)
	}

	// Zero-length axis: scans allow it and return an equal-shaped empty result.
	z, _ := New(0, 3)
	zs, err := z.CumSum(0)
	if err != nil {
		t.Fatalf("CumSum zero-axis: %v", err)
	}
	wantData(t, zs, []int{0, 3}, []float64{})
}

func TestClip(t *testing.T) {
	a := mat(t)
	c, _ := a.Clip(2, 7)
	wantData(t, c, []int{2, 3}, []float64{2, 5, 3, 7, 2, 7})

	if _, err := a.Clip(5, 1); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Clip lo>hi err = %v", err)
	}
}

func TestWhere(t *testing.T) {
	cond, _ := FromData([]float64{1, 0, 1, 0, 1, 0}, 2, 3)
	tv, _ := Full(10, 2, 3)
	fv, _ := Full(-1, 2, 3)
	w, err := Where(cond, tv, fv)
	if err != nil {
		t.Fatal(err)
	}
	wantData(t, w, []int{2, 3}, []float64{10, -1, 10, -1, 10, -1})

	// Broadcasting across all three operands: (3,) cond, (1,) t, (2,1) f.
	c2, _ := FromData([]float64{1, 0, 1}, 3)
	t2, _ := FromData([]float64{5}, 1)
	f2, _ := FromData([]float64{1, 2}, 2, 1)
	wb, err := Where(c2, t2, f2)
	if err != nil {
		t.Fatal(err)
	}
	wantData(t, wb, []int{2, 3}, []float64{5, 1, 5, 5, 2, 5})

	// Non-broadcastable shapes error (cond vs t, then the folded shape vs f).
	bad, _ := FromData([]float64{1, 2}, 2)
	if _, err := Where(cond, bad, fv); !errors.Is(err, ErrBroadcast) {
		t.Errorf("Where cond/t mismatch err = %v", err)
	}
	if _, err := Where(cond, tv, bad); !errors.Is(err, ErrBroadcast) {
		t.Errorf("Where folded/f mismatch err = %v", err)
	}
}
