package ndarray

import (
	"errors"
	"testing"
)

func TestLinspace(t *testing.T) {
	// numpy.linspace, default endpoint=True.
	a := mustArr(t, ok(Linspace(0, 1, 5)))
	eqFloats(t, a.materialize(), []float64{0, 0.25, 0.5, 0.75, 1})

	b := mustArr(t, ok(Linspace(0, 10, 11)))
	eqFloats(t, b.materialize(),
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})

	one := mustArr(t, ok(Linspace(2, 3, 1)))
	eqFloats(t, one.materialize(), []float64{2})

	zero := mustArr(t, ok(Linspace(0, 1, 0)))
	if zero.Size() != 0 {
		t.Fatalf("linspace 0 size %d", zero.Size())
	}

	if _, err := Linspace(0, 1, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("negative num: %v", err)
	}
}

func TestEyeAndIdentity(t *testing.T) {
	// numpy.eye(3).
	e := mustArr(t, ok(Eye(3, 0, 0)))
	eqInts(t, e.Shape(), []int{3, 3})
	eqFloats(t, e.materialize(),
		[]float64{1, 0, 0, 0, 1, 0, 0, 0, 1})

	// numpy.eye(2,3).
	r := mustArr(t, ok(Eye(2, 3, 0)))
	eqInts(t, r.Shape(), []int{2, 3})
	eqFloats(t, r.materialize(), []float64{1, 0, 0, 0, 1, 0})

	// numpy.eye(3, k=1) superdiagonal.
	sup := mustArr(t, ok(Eye(3, 0, 1)))
	eqFloats(t, sup.materialize(),
		[]float64{0, 1, 0, 0, 0, 1, 0, 0, 0})

	// numpy.eye(3, k=-1) subdiagonal.
	sub := mustArr(t, ok(Eye(3, 0, -1)))
	eqFloats(t, sub.materialize(),
		[]float64{0, 0, 0, 1, 0, 0, 0, 1, 0})

	// numpy.eye(2,4,k=2): offset beyond the last column is dropped.
	off := mustArr(t, ok(Eye(2, 4, 2)))
	eqInts(t, off.Shape(), []int{2, 4})
	eqFloats(t, off.materialize(),
		[]float64{0, 0, 1, 0, 0, 0, 0, 1})

	id := mustArr(t, ok(Identity(2)))
	eqFloats(t, id.materialize(), []float64{1, 0, 0, 1})

	if _, err := Eye(-1, 0, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("negative eye: %v", err)
	}
	// A non-positive m defaults to n (square); Eye(2,-3,0) == Eye(2,2,0).
	sq := mustArr(t, ok(Eye(2, -3, 0)))
	eqInts(t, sq.Shape(), []int{2, 2})
}

func TestReshapeInfer(t *testing.T) {
	a := mustArr(t, ok(Arange(0, 6, 1)))
	// reshape(2,-1) -> (2,3); reshape(-1,3) -> (2,3); reshape(-1) -> (6,).
	for _, tc := range []struct {
		in   []int
		want []int
	}{
		{[]int{2, -1}, []int{2, 3}},
		{[]int{-1, 3}, []int{2, 3}},
		{[]int{-1}, []int{6}},
		{[]int{3, -1, 1}, []int{3, 2, 1}},
	} {
		r := mustArr(t, ok(a.Reshape(tc.in...)))
		eqInts(t, r.Shape(), tc.want)
	}

	if _, err := a.Reshape(-1, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("two -1: %v", err)
	}
	if _, err := a.Reshape(4, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("indivisible -1: %v", err)
	}
	// -1 with a zero known dimension cannot be inferred.
	z := mustArr(t, ok(New(0)))
	if _, err := z.Reshape(0, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("zero-known -1: %v", err)
	}
}
