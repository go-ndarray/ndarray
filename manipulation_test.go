package ndarray

import (
	"errors"
	"testing"
)

func TestFlatten(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	f := a.Flatten()
	eqInts(t, f.Shape(), []int{6})
	eqFloats(t, f.materialize(), []float64{1, 2, 3, 4, 5, 6})
	// Flatten is a copy: mutating it leaves the source untouched.
	f.Set(99, 0)
	if a.At(0, 0) != 1 {
		t.Fatalf("Flatten should copy; a[0,0]=%v", a.At(0, 0))
	}
	// Flatten of a transposed (strided) view is row-major of the view.
	tr := a.Transpose().Flatten()
	eqFloats(t, tr.materialize(), []float64{1, 4, 2, 5, 3, 6})
}

func TestExpandDims(t *testing.T) {
	v := mustArr(t, ok(FromData([]float64{0, 1, 2}, 3)))
	e0 := mustArr(t, ok(v.ExpandDims(0)))
	eqInts(t, e0.Shape(), []int{1, 3})
	e1 := mustArr(t, ok(v.ExpandDims(1)))
	eqInts(t, e1.Shape(), []int{3, 1})
	// Negative axis: -1 inserts at the end.
	eNeg := mustArr(t, ok(v.ExpandDims(-1)))
	eqInts(t, eNeg.Shape(), []int{3, 1})
	eqFloats(t, e1.materialize(), []float64{0, 1, 2})

	if _, err := v.ExpandDims(3); !errors.Is(err, ErrAxis) {
		t.Fatalf("expand bad axis: %v", err)
	}
}

func TestSqueeze(t *testing.T) {
	s := mustArr(t, ok(FromData([]float64{0, 1, 2}, 1, 3, 1)))
	all := mustArr(t, ok(s.Squeeze()))
	eqInts(t, all.Shape(), []int{3})

	ax0 := mustArr(t, ok(s.Squeeze(0)))
	eqInts(t, ax0.Shape(), []int{3, 1})

	axNeg := mustArr(t, ok(s.Squeeze(-1)))
	eqInts(t, axNeg.Shape(), []int{1, 3})

	if _, err := s.Squeeze(1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("squeeze non-1 axis: %v", err)
	}
	if _, err := s.Squeeze(5); !errors.Is(err, ErrAxis) {
		t.Fatalf("squeeze bad axis: %v", err)
	}
}

func TestConcatenate(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	b := mustArr(t, ok(FromData([]float64{6, 7, 8, 9, 10, 11}, 2, 3)))

	c0 := mustArr(t, ok(Concatenate([]*Array{a, b}, 0)))
	eqInts(t, c0.Shape(), []int{4, 3})
	eqFloats(t, c0.materialize(),
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})

	c1 := mustArr(t, ok(Concatenate([]*Array{a, b}, 1)))
	eqInts(t, c1.Shape(), []int{2, 6})
	eqFloats(t, c1.materialize(),
		[]float64{0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11})

	// Negative axis.
	cNeg := mustArr(t, ok(Concatenate([]*Array{a, b}, -1)))
	eqInts(t, cNeg.Shape(), []int{2, 6})

	// 1-D.
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	w := mustArr(t, ok(FromData([]float64{4, 5, 6}, 3)))
	c1d := mustArr(t, ok(Concatenate([]*Array{u, w}, 0)))
	eqFloats(t, c1d.materialize(), []float64{1, 2, 3, 4, 5, 6})
}

func TestConcatenateErrors(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	if _, err := Concatenate(nil, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty concat: %v", err)
	}
	if _, err := Concatenate([]*Array{a}, 5); !errors.Is(err, ErrAxis) {
		t.Fatalf("concat bad axis: %v", err)
	}
	// Rank mismatch.
	v := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	if _, err := Concatenate([]*Array{a, v}, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("concat rank mismatch: %v", err)
	}
	// Non-axis shape mismatch.
	bad := mustArr(t, ok(FromData([]float64{0, 1, 2, 3}, 2, 2)))
	if _, err := Concatenate([]*Array{a, bad}, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("concat shape mismatch: %v", err)
	}
}

func TestStack(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	b := mustArr(t, ok(FromData([]float64{6, 7, 8, 9, 10, 11}, 2, 3)))

	s0 := mustArr(t, ok(Stack([]*Array{a, b}, 0)))
	eqInts(t, s0.Shape(), []int{2, 2, 3})
	eqFloats(t, s0.materialize(),
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})

	s1 := mustArr(t, ok(Stack([]*Array{a, b}, 1)))
	eqInts(t, s1.Shape(), []int{2, 2, 3})
	eqFloats(t, s1.materialize(),
		[]float64{0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11})

	s2 := mustArr(t, ok(Stack([]*Array{a, b}, 2)))
	eqInts(t, s2.Shape(), []int{2, 3, 2})
	eqFloats(t, s2.materialize(),
		[]float64{0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11})

	// Negative axis: -1 inserts the new axis at the end (same as axis 2 here).
	sNeg := mustArr(t, ok(Stack([]*Array{a, b}, -1)))
	eqInts(t, sNeg.Shape(), []int{2, 3, 2})
	eqFloats(t, sNeg.materialize(),
		[]float64{0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11})
}

func TestStackErrors(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	if _, err := Stack(nil, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty stack: %v", err)
	}
	// Different rank.
	bad := mustArr(t, ok(FromData([]float64{0, 1}, 2)))
	if _, err := Stack([]*Array{a, bad}, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("stack rank mismatch: %v", err)
	}
	// Same rank, different shape: exercises sameShape's element comparison.
	bad2 := mustArr(t, ok(FromData([]float64{0, 1, 2, 3}, 2, 2)))
	if _, err := Stack([]*Array{a, bad2}, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("stack shape mismatch: %v", err)
	}
	if _, err := Stack([]*Array{a}, 5); !errors.Is(err, ErrAxis) {
		t.Fatalf("stack bad axis: %v", err)
	}
}

func TestVStackHStack(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	b := mustArr(t, ok(FromData([]float64{6, 7, 8, 9, 10, 11}, 2, 3)))

	v := mustArr(t, ok(VStack([]*Array{a, b})))
	eqInts(t, v.Shape(), []int{4, 3})
	eqFloats(t, v.materialize(),
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})

	h := mustArr(t, ok(HStack([]*Array{a, b})))
	eqInts(t, h.Shape(), []int{2, 6})
	eqFloats(t, h.materialize(),
		[]float64{0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11})

	// 1-D: vstack makes rows (2,3); hstack concatenates to (6,).
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	w := mustArr(t, ok(FromData([]float64{4, 5, 6}, 3)))
	v1 := mustArr(t, ok(VStack([]*Array{u, w})))
	eqInts(t, v1.Shape(), []int{2, 3})
	eqFloats(t, v1.materialize(), []float64{1, 2, 3, 4, 5, 6})
	h1 := mustArr(t, ok(HStack([]*Array{u, w})))
	eqInts(t, h1.Shape(), []int{6})
	eqFloats(t, h1.materialize(), []float64{1, 2, 3, 4, 5, 6})

	if _, err := VStack(nil); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty vstack: %v", err)
	}
	if _, err := HStack(nil); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty hstack: %v", err)
	}
}
