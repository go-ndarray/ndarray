package ndarray

import (
	"math"
	"reflect"
	"testing"
)

// wantNaNData is wantData with NaN == NaN, for the NaN-propagation tables.
func wantNaNData(t *testing.T, name string, a *Array, shape []int, data []float64) {
	t.Helper()
	if !reflect.DeepEqual(a.Shape(), shape) {
		t.Fatalf("%s: shape = %v, want %v", name, a.Shape(), shape)
	}
	got := a.materialize()
	for i := range data {
		if got[i] != data[i] && !(math.IsNaN(got[i]) && math.IsNaN(data[i])) {
			t.Fatalf("%s: data = %v, want %v", name, got, data)
		}
	}
}

// TestNaNPropagation locks the NaN convention of the per-axis and index
// reductions and of the pairwise Maximum/Minimum to numpy's: a NaN wins. The
// expectations are numpy 2.5.3 values for the same fixture. Before the fix the
// per-axis Max/Min and every Arg* reduction skipped a NaN that was not the first
// element (and stuck to one that was), and Maximum/Minimum returned the
// infinity when the other operand was NaN (math.Max/math.Min special-case ±Inf
// ahead of NaN).
func TestNaNPropagation(t *testing.T) {
	nan := math.NaN()
	a := mustArr(t, ok(FromData([]float64{
		1, nan, 3, 5,
		9, 4, nan, 7,
		nan, 8, 2, 6,
	}, 3, 4)))

	if got, _ := a.ArgMax(); got != 1 {
		t.Errorf("ArgMax = %d, want 1 (first NaN)", got)
	}
	if got, _ := a.ArgMin(); got != 1 {
		t.Errorf("ArgMin = %d, want 1 (first NaN)", got)
	}
	inf := math.Inf(1)
	for _, c := range []struct {
		data       []float64
		argmax, am int
	}{
		{[]float64{inf, nan}, 1, 1},
		{[]float64{-inf, nan}, 1, 1},
		{[]float64{nan, nan, 5}, 0, 0},
	} {
		v := mustArr(t, ok(FromData(c.data, len(c.data))))
		if got, _ := v.ArgMax(); got != c.argmax {
			t.Errorf("ArgMax(%v) = %d, want %d", c.data, got, c.argmax)
		}
		if got, _ := v.ArgMin(); got != c.am {
			t.Errorf("ArgMin(%v) = %d, want %d", c.data, got, c.am)
		}
	}

	for _, c := range []struct {
		name  string
		fn    func(int, bool) (*Array, error)
		axis  int
		shape []int
		data  []float64
	}{
		{"max0", a.MaxAxis, 0, []int{4}, []float64{nan, nan, nan, 7}},
		{"min0", a.MinAxis, 0, []int{4}, []float64{nan, nan, nan, 5}},
		{"max1", a.MaxAxis, 1, []int{3}, []float64{nan, nan, nan}},
		{"min1", a.MinAxis, 1, []int{3}, []float64{nan, nan, nan}},
		{"argmax0", a.ArgMaxAxis, 0, []int{4}, []float64{2, 0, 1, 1}},
		{"argmin0", a.ArgMinAxis, 0, []int{4}, []float64{2, 0, 1, 0}},
		{"argmax1", a.ArgMaxAxis, 1, []int{3}, []float64{1, 2, 0}},
		{"argmin1", a.ArgMinAxis, 1, []int{3}, []float64{1, 2, 0}},
	} {
		got, err := c.fn(c.axis, false)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		wantNaNData(t, c.name, got, c.shape, c.data)
	}

	x := mustArr(t, ok(FromData([]float64{nan, nan, 1, -inf, nan, 2}, 6)))
	y := mustArr(t, ok(FromData([]float64{inf, -inf, nan, nan, nan, 1}, 6)))
	mx := mustArr(t, ok(x.Maximum(y)))
	wantNaNData(t, "maximum", mx, []int{6}, []float64{nan, nan, nan, nan, nan, 2})
	mn := mustArr(t, ok(x.Minimum(y)))
	wantNaNData(t, "minimum", mn, []int{6}, []float64{nan, nan, nan, nan, nan, 1})
}
