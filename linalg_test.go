package ndarray

import (
	"errors"
	"testing"
)

func TestMatMul(t *testing.T) {
	// A (2x3) @ B (3x2), numpy 2.2.4.
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	b := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 3, 2)))
	r := mustArr(t, ok(a.MatMul(b)))
	eqInts(t, r.Shape(), []int{2, 2})
	eqFloats(t, r.materialize(), []float64{10, 13, 28, 40})

	// Non-square (3x2)@(2x3) -> (3x3).
	A := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 3, 2)))
	B := mustArr(t, ok(FromData([]float64{7, 8, 9, 10, 11, 12}, 2, 3)))
	rr := mustArr(t, ok(A.MatMul(B)))
	eqInts(t, rr.Shape(), []int{3, 3})
	eqFloats(t, rr.materialize(),
		[]float64{27, 30, 33, 61, 68, 75, 95, 106, 117})
}

func TestMatMulTransposedView(t *testing.T) {
	// A.T @ A where A is (3x2): A.T is a (2x3) strided view, exercising
	// materialize inside matmul2D. Result (2x2) == [[35,44],[44,56]] (numpy).
	A := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 3, 2)))
	r := mustArr(t, ok(A.Transpose().MatMul(A)))
	eqInts(t, r.Shape(), []int{2, 2})
	eqFloats(t, r.materialize(), []float64{35, 44, 44, 56})
}

func TestMatMulErrors(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	v := mustArr(t, ok(FromData([]float64{1, 2}, 2)))
	if _, err := a.MatMul(v); !errors.Is(err, ErrLinalg) {
		t.Fatalf("non-2D operand: %v", err)
	}
	bad := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 3, 2)))
	if _, err := a.MatMul(bad); !errors.Is(err, ErrLinalg) {
		t.Fatalf("inner-dim mismatch: %v", err)
	}
}

func TestDot(t *testing.T) {
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	v := mustArr(t, ok(FromData([]float64{4, 5, 6}, 3)))

	// 1-D · 1-D -> scalar (0-d) 32.
	s := mustArr(t, ok(u.Dot(v)))
	if s.Ndim() != 0 || s.Size() != 1 {
		t.Fatalf("dot1d Ndim=%d Size=%d", s.Ndim(), s.Size())
	}
	eqFloats(t, s.materialize(), []float64{32})

	// 2-D · 2-D -> matmul.
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	b := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 3, 2)))
	mm := mustArr(t, ok(a.Dot(b)))
	eqFloats(t, mm.materialize(), []float64{10, 13, 28, 40})

	// 2-D · 1-D -> matrix-vector: A·[1,1,1] = [3,12].
	ones := mustArr(t, ok(FromData([]float64{1, 1, 1}, 3)))
	mv := mustArr(t, ok(a.Dot(ones)))
	eqInts(t, mv.Shape(), []int{2})
	eqFloats(t, mv.materialize(), []float64{3, 12})

	// 1-D · 2-D -> vector-matrix: [1,1]·A = [3,5,7].
	ones2 := mustArr(t, ok(FromData([]float64{1, 1}, 2)))
	vm := mustArr(t, ok(ones2.Dot(a)))
	eqInts(t, vm.Shape(), []int{3})
	eqFloats(t, vm.materialize(), []float64{3, 5, 7})
}

func TestDotErrors(t *testing.T) {
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	w := mustArr(t, ok(FromData([]float64{1, 2}, 2)))
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))

	if _, err := u.Dot(w); !errors.Is(err, ErrLinalg) {
		t.Fatalf("dot 1d length mismatch: %v", err)
	}
	if _, err := a.Dot(w); !errors.Is(err, ErrLinalg) {
		t.Fatalf("dot 2d-1d mismatch: %v", err)
	}
	// 1-D · 2-D: vector length must equal the matrix's first dim (a has 2 rows).
	if _, err := u.Dot(a); !errors.Is(err, ErrLinalg) {
		t.Fatalf("dot 1d-2d mismatch: %v", err)
	}
	// 3-D operand is unsupported.
	c := mustArr(t, ok(New(2, 2, 2)))
	if _, err := c.Dot(u); !errors.Is(err, ErrLinalg) {
		t.Fatalf("dot high-rank: %v", err)
	}
}

func TestInner(t *testing.T) {
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	v := mustArr(t, ok(FromData([]float64{4, 5, 6}, 3)))
	s := mustArr(t, ok(u.Inner(v)))
	eqFloats(t, s.materialize(), []float64{32})

	// 2-D inner: inner(A, C) sums over the last axis. A (2x3), C (2x3) ->
	// (2x2). numpy.inner(arange(6).reshape(2,3), [[0,2,4],[1,3,5]]).
	a := mustArr(t, ok(FromData([]float64{0, 1, 2, 3, 4, 5}, 2, 3)))
	c := mustArr(t, ok(FromData([]float64{0, 2, 4, 1, 3, 5}, 2, 3)))
	r := mustArr(t, ok(a.Inner(c)))
	eqInts(t, r.Shape(), []int{2, 2})
	eqFloats(t, r.materialize(), []float64{10, 13, 28, 40})

	if _, err := a.Inner(mustArr(t, ok(FromData([]float64{1, 2}, 1, 2)))); !errors.Is(err, ErrLinalg) {
		t.Fatalf("inner last-dim mismatch: %v", err)
	}
	if _, err := a.Inner(u); !errors.Is(err, ErrLinalg) {
		t.Fatalf("inner rank mismatch: %v", err)
	}
}

func TestOuter(t *testing.T) {
	u := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	v := mustArr(t, ok(FromData([]float64{4, 5, 6}, 3)))
	o := u.Outer(v)
	eqInts(t, o.Shape(), []int{3, 3})
	eqFloats(t, o.materialize(),
		[]float64{4, 5, 6, 8, 10, 12, 12, 15, 18})

	// Outer flattens higher-rank inputs first: (2x2) and (2,) -> (4x2).
	m := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	o2 := m.Outer(mustArr(t, ok(FromData([]float64{1, 10}, 2))))
	eqInts(t, o2.Shape(), []int{4, 2})
	eqFloats(t, o2.materialize(),
		[]float64{1, 10, 2, 20, 3, 30, 4, 40})
}
