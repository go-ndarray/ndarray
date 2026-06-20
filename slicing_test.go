package ndarray

import (
	"errors"
	"testing"
)

// arange24 builds arange(24).reshape(2,3,4) for slicing tests.
func arange24(t *testing.T) *Array {
	t.Helper()
	data := make([]float64, 24)
	for i := range data {
		data[i] = float64(i)
	}
	return mustArr(t, ok(FromData(data, 2, 3, 4)))
}

func TestSliceIntegerIndexDropsAxis(t *testing.T) {
	a := arange24(t)
	// a[1] -> shape (3,4), numpy 2.2.4.
	r := mustArr(t, ok(a.Slice(A(1), All(), All())))
	eqInts(t, r.Shape(), []int{3, 4})
	eqFloats(t, r.materialize(),
		[]float64{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})

	// a[:,1,:] -> shape (2,4).
	r2 := mustArr(t, ok(a.Slice(All(), A(1), All())))
	eqInts(t, r2.Shape(), []int{2, 4})
	eqFloats(t, r2.materialize(), []float64{4, 5, 6, 7, 16, 17, 18, 19})
}

func TestSliceStepFull(t *testing.T) {
	a := arange24(t)
	// a[0, 1:3, ::2] -> [[4,6],[8,10]], shape (2,2).
	r := mustArr(t, ok(a.Slice(A(0), R(1, 3), Step(2))))
	eqInts(t, r.Shape(), []int{2, 2})
	eqFloats(t, r.materialize(), []float64{4, 6, 8, 10})
}

func TestSliceNegativeStepReverse(t *testing.T) {
	m := mustArr(t, ok(FromData(
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)))
	// m[::-1] reverses rows.
	rows := mustArr(t, ok(m.Slice(Step(-1), All())))
	eqInts(t, rows.Shape(), []int{3, 4})
	eqFloats(t, rows.materialize(),
		[]float64{8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3})

	// m[:, ::-1] reverses columns.
	cols := mustArr(t, ok(m.Slice(All(), Step(-1))))
	eqFloats(t, cols.materialize(),
		[]float64{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8})
}

func TestSliceNegativeBounds(t *testing.T) {
	m := mustArr(t, ok(FromData(
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)))
	// m[1:, 1:3] -> [[5,6],[9,10]].
	r := mustArr(t, ok(m.Slice(From(1), R(1, 3))))
	eqInts(t, r.Shape(), []int{2, 2})
	eqFloats(t, r.materialize(), []float64{5, 6, 9, 10})

	// m[-2:, -1] -> [7, 11], integer index on the last axis drops it.
	r2 := mustArr(t, ok(m.Slice(From(-2), A(-1))))
	eqInts(t, r2.Shape(), []int{2})
	eqFloats(t, r2.materialize(), []float64{7, 11})
}

func TestSlice1DVariants(t *testing.T) {
	v := mustArr(t, ok(Arange(0, 10, 1)))
	cases := []struct {
		name string
		ix   Index
		want []float64
	}{
		{"2:8:3", Rng(2, 8, 3), []float64{2, 5}},
		{"8:2:-2", Rng(8, 2, -2), []float64{8, 6, 4}},
		{"::-1", Step(-1), []float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
		{"100:", From(100), nil},
		{"-3:", From(-3), []float64{7, 8, 9}},
		{":-3", To(-3), []float64{0, 1, 2, 3, 4, 5, 6}},
		{":100", To(100), []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},
		{"20:", From(20), nil},
		// 20::-1 : start clamps to length-1, full reverse (start-only, neg step).
		{"20::-1", Index{start: 20, step: -1, hasStart: true},
			[]float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
		// :-100:-1 : stop clamps to -1, full reverse (stop-only, neg step).
		{":-100:-1", Index{stop: -100, step: -1, hasStop: true},
			[]float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := mustArr(t, ok(v.Slice(c.ix)))
			if c.want == nil {
				if r.Size() != 0 {
					t.Fatalf("want empty, got %v", r.materialize())
				}
				return
			}
			eqFloats(t, r.materialize(), c.want)
		})
	}
}

func TestSliceIsViewWriteThrough(t *testing.T) {
	m := mustArr(t, ok(FromData(
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)))
	v := mustArr(t, ok(m.Slice(From(1), R(1, 3)))) // m[1:,1:3], shares data
	v.Set(99, 0, 0)                                // == m[1,1]
	if m.At(1, 1) != 99 {
		t.Fatalf("slice should be a view; m[1,1]=%v", m.At(1, 1))
	}
	// And a write to the original is visible through the view.
	m.Set(42, 2, 2)
	if v.At(1, 1) != 42 {
		t.Fatalf("view should see writes to source; v[1,1]=%v", v.At(1, 1))
	}
}

func TestSliceErrors(t *testing.T) {
	m := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	if _, err := m.Slice(A(0)); !errors.Is(err, ErrIndex) {
		t.Fatalf("too few indices: %v", err)
	}
	if _, err := m.Slice(A(5), All()); !errors.Is(err, ErrIndex) {
		t.Fatalf("int index out of range: %v", err)
	}
	if _, err := m.Slice(A(-5), All()); !errors.Is(err, ErrIndex) {
		t.Fatalf("negative int index out of range: %v", err)
	}
	if _, err := m.Slice(Rng(0, 2, 0), All()); !errors.Is(err, ErrIndex) {
		t.Fatalf("zero step: %v", err)
	}
}
