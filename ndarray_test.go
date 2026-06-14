package ndarray

import (
	"errors"
	"math"
	"testing"
)

// res bundles a constructor's (*Array, error) so it can be passed to mustArr
// in a single-value context.
type res struct {
	a   *Array
	err error
}

func mustArr(t *testing.T, r res) *Array {
	t.Helper()
	if r.err != nil {
		t.Fatalf("unexpected error: %v", r.err)
	}
	return r.a
}

func ok(a *Array, err error) res { return res{a, err} }

func eqInts(t *testing.T, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
}

func eqFloats(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d] = %v, want %v (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
}

func TestNewZerosOnesFull(t *testing.T) {
	a := mustArr(t, ok(New(2, 3)))
	eqInts(t, a.Shape(), []int{2, 3})
	if a.Size() != 6 || a.Ndim() != 2 {
		t.Fatalf("Size=%d Ndim=%d", a.Size(), a.Ndim())
	}
	eqFloats(t, a.Ravel().materialize(), []float64{0, 0, 0, 0, 0, 0})

	z := mustArr(t, ok(Zeros(2)))
	eqFloats(t, z.materialize(), []float64{0, 0})

	o := mustArr(t, ok(Ones(3)))
	eqFloats(t, o.materialize(), []float64{1, 1, 1})

	f := mustArr(t, ok(Full(2.5, 2, 2)))
	eqFloats(t, f.materialize(), []float64{2.5, 2.5, 2.5, 2.5})

	// scalar (0-d) array
	s := mustArr(t, ok(New()))
	if s.Size() != 1 || s.Ndim() != 0 {
		t.Fatalf("scalar Size=%d Ndim=%d", s.Size(), s.Ndim())
	}
}

func TestConstructorErrors(t *testing.T) {
	if _, err := New(2, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("New negative: %v", err)
	}
	if _, err := Zeros(-1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("Zeros negative: %v", err)
	}
	if _, err := Ones(-1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("Ones negative: %v", err)
	}
	if _, err := Full(1, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("Full negative: %v", err)
	}
	if _, err := FromData([]float64{1, 2}, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("FromData negative: %v", err)
	}
	if _, err := FromData([]float64{1, 2, 3}, 2, 2); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("FromData size mismatch: %v", err)
	}
}

func TestFromDataCopies(t *testing.T) {
	src := []float64{1, 2, 3, 4}
	a := mustArr(t, ok(FromData(src, 2, 2)))
	src[0] = 99
	if a.At(0, 0) != 1 {
		t.Fatalf("FromData did not copy: At(0,0)=%v", a.At(0, 0))
	}
}

func TestArange(t *testing.T) {
	a := mustArr(t, ok(Arange(0, 5, 1)))
	eqFloats(t, a.materialize(), []float64{0, 1, 2, 3, 4})

	b := mustArr(t, ok(Arange(1, 2, 0.5)))
	eqFloats(t, b.materialize(), []float64{1, 1.5})

	c := mustArr(t, ok(Arange(5, 0, -2)))
	eqFloats(t, c.materialize(), []float64{5, 3, 1})

	empty := mustArr(t, ok(Arange(0, 0, 1)))
	if empty.Size() != 0 {
		t.Fatalf("empty arange size %d", empty.Size())
	}

	if _, err := Arange(0, 1, 0); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("zero step: %v", err)
	}
}

func TestAtSet(t *testing.T) {
	a := mustArr(t, ok(New(2, 3)))
	a.Set(7, 1, 2)
	if a.At(1, 2) != 7 {
		t.Fatalf("At = %v, want 7", a.At(1, 2))
	}
}

func TestAtPanics(t *testing.T) {
	a := mustArr(t, ok(New(2, 2)))
	cases := []struct {
		name string
		idx  []int
	}{
		{"rank", []int{0}},
		{"high", []int{2, 0}},
		{"low", []int{0, -1}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("expected panic for %v", c.idx)
				}
			}()
			a.At(c.idx...)
		})
	}
}

func TestReshape(t *testing.T) {
	a := mustArr(t, ok(Arange(0, 6, 1)))
	r := mustArr(t, ok(a.Reshape(2, 3)))
	eqInts(t, r.Shape(), []int{2, 3})
	if r.At(1, 1) != 4 {
		t.Fatalf("reshape At(1,1)=%v want 4", r.At(1, 1))
	}

	// reshape of a non-contiguous (transposed) array goes through materialize.
	tr := r.Transpose()
	r2 := mustArr(t, ok(tr.Reshape(6)))
	eqFloats(t, r2.materialize(), []float64{0, 3, 1, 4, 2, 5})

	if _, err := a.Reshape(4); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("bad reshape: %v", err)
	}
	if _, err := a.Reshape(-1); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("negative reshape: %v", err)
	}
}

func TestRavelAndTranspose(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	tr := a.Transpose()
	eqInts(t, tr.Shape(), []int{3, 2})
	if tr.At(0, 1) != 4 {
		t.Fatalf("transpose At(0,1)=%v want 4", tr.At(0, 1))
	}
	eqFloats(t, tr.Ravel().materialize(), []float64{1, 4, 2, 5, 3, 6})
}

func TestCopyIndependent(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	c := a.Copy()
	c.Set(99, 0, 0)
	if a.At(0, 0) != 1 {
		t.Fatalf("Copy not independent: a changed to %v", a.At(0, 0))
	}
	// Copy of a transposed view yields a contiguous array.
	ct := a.Transpose().Copy()
	eqFloats(t, ct.materialize(), []float64{1, 3, 2, 4})
}

func TestString(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2.5, 3}, 3)))
	got := a.String()
	want := "Array(shape=[3], data=[1 2.5 3])"
	if got != want {
		t.Fatalf("String = %q, want %q", got, want)
	}
	// multi-dim shape, exercise the space separators in shape.
	b := mustArr(t, ok(New(2, 2)))
	wantB := "Array(shape=[2 2], data=[0 0 0 0])"
	if b.String() != wantB {
		t.Fatalf("String = %q, want %q", b.String(), wantB)
	}
}

func TestBinOpSameShape(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	b := mustArr(t, ok(FromData([]float64{5, 6, 7, 8}, 2, 2)))

	add := mustArr(t, ok(a.Add(b)))
	eqFloats(t, add.materialize(), []float64{6, 8, 10, 12})
	sub := mustArr(t, ok(a.Sub(b)))
	eqFloats(t, sub.materialize(), []float64{-4, -4, -4, -4})
	mul := mustArr(t, ok(a.Mul(b)))
	eqFloats(t, mul.materialize(), []float64{5, 12, 21, 32})
	div := mustArr(t, ok(b.Div(a)))
	eqFloats(t, div.materialize(), []float64{5, 3, 7.0 / 3.0, 2})
}

func TestBroadcasting(t *testing.T) {
	// (2,3) + (3,)  -> row vector broadcast over rows
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	row := mustArr(t, ok(FromData([]float64{10, 20, 30}, 3)))
	r := mustArr(t, ok(a.Add(row)))
	eqFloats(t, r.materialize(), []float64{11, 22, 33, 14, 25, 36})

	// (2,3) + (2,1) -> column vector broadcast over columns
	col := mustArr(t, ok(FromData([]float64{100, 200}, 2, 1)))
	r2 := mustArr(t, ok(a.Add(col)))
	eqFloats(t, r2.materialize(), []float64{101, 102, 103, 204, 205, 206})

	// (1,3) + (2,1) -> (2,3): both operands broadcast
	x := mustArr(t, ok(FromData([]float64{1, 2, 3}, 1, 3)))
	y := mustArr(t, ok(FromData([]float64{10, 20}, 2, 1)))
	r3 := mustArr(t, ok(x.Add(y)))
	eqInts(t, r3.Shape(), []int{2, 3})
	eqFloats(t, r3.materialize(), []float64{11, 12, 13, 21, 22, 23})

	// incompatible
	bad := mustArr(t, ok(FromData([]float64{1, 2}, 2)))
	if _, err := a.Add(bad); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected broadcast error, got %v", err)
	}
}

func TestBroadcastEmpty(t *testing.T) {
	// A zero-size array broadcast keeps zero size and exercises the empty
	// fast-path in broadcastTo.
	a := mustArr(t, ok(New(0, 3)))
	b := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	r := mustArr(t, ok(a.Add(b)))
	if r.Size() != 0 {
		t.Fatalf("empty broadcast size %d", r.Size())
	}
	eqInts(t, r.Shape(), []int{0, 3})
}

func TestScalarOps(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	eqFloats(t, a.AddScalar(10).materialize(), []float64{11, 12, 13, 14})
	eqFloats(t, a.SubScalar(1).materialize(), []float64{0, 1, 2, 3})
	eqFloats(t, a.MulScalar(2).materialize(), []float64{2, 4, 6, 8})
	eqFloats(t, a.DivScalar(2).materialize(), []float64{0.5, 1, 1.5, 2})
}

func TestMapNegAbs(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, -2, 3, -4}, 2, 2)))
	eqFloats(t, a.Map(func(x float64) float64 { return x + 1 }).materialize(),
		[]float64{2, -1, 4, -3})
	eqFloats(t, a.Neg().materialize(), []float64{-1, 2, -3, 4})
	eqFloats(t, a.Abs().materialize(), []float64{1, 2, 3, 4})
}

func TestReductions(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	if a.Sum() != 10 {
		t.Fatalf("Sum = %v", a.Sum())
	}
	if a.Prod() != 24 {
		t.Fatalf("Prod = %v", a.Prod())
	}
	mean, err := a.Mean()
	if err != nil || mean != 2.5 {
		t.Fatalf("Mean = %v, err %v", mean, err)
	}
	mx, err := a.Max()
	if err != nil || mx != 4 {
		t.Fatalf("Max = %v, err %v", mx, err)
	}
	mn, err := a.Min()
	if err != nil || mn != 1 {
		t.Fatalf("Min = %v, err %v", mn, err)
	}
}

func TestReductionEmptyErrors(t *testing.T) {
	e := mustArr(t, ok(New(0)))
	if e.Sum() != 0 {
		t.Fatalf("empty Sum = %v", e.Sum())
	}
	if e.Prod() != 1 {
		t.Fatalf("empty Prod = %v", e.Prod())
	}
	if _, err := e.Mean(); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty Mean: %v", err)
	}
	if _, err := e.Max(); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty Max: %v", err)
	}
	if _, err := e.Min(); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("empty Min: %v", err)
	}
}

func TestForEachEmpty(t *testing.T) {
	// materialize on an empty array hits the n==0 early return in forEach.
	e := mustArr(t, ok(New(0, 4)))
	if len(e.materialize()) != 0 {
		t.Fatalf("expected empty materialize")
	}
}

func TestContiguousReshapeSharesData(t *testing.T) {
	a := mustArr(t, ok(Arange(0, 6, 1)))
	r := mustArr(t, ok(a.Reshape(2, 3)))
	r.Set(99, 0, 0)
	// contiguous reshape shares backing data, so the source sees the write.
	if a.At(0) != 99 {
		t.Fatalf("contiguous reshape should share data; a[0]=%v", a.At(0))
	}
}

func TestBroadcastTrailingOne(t *testing.T) {
	// (2,3) + (1,): trailing dim of the right operand is 1 (d2 == 1 branch).
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	one := mustArr(t, ok(FromData([]float64{10}, 1)))
	r := mustArr(t, ok(a.Add(one)))
	eqInts(t, r.Shape(), []int{2, 3})
	eqFloats(t, r.materialize(), []float64{11, 12, 13, 14, 15, 16})
}

func TestBroadcastLeftShorter(t *testing.T) {
	// (3,) + (2,3): the left operand has fewer dims, exercising the j<0 branch
	// for s1 in broadcastShape (and a row-vector broadcast over rows).
	row := mustArr(t, ok(FromData([]float64{10, 20, 30}, 3)))
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	r := mustArr(t, ok(row.Add(a)))
	eqInts(t, r.Shape(), []int{2, 3})
	eqFloats(t, r.materialize(), []float64{11, 22, 33, 14, 25, 36})
}

func TestIsContiguousOffset(t *testing.T) {
	// An array with a non-zero offset is not contiguous; Reshape must copy.
	v := &Array{
		data:    []float64{0, 1, 2, 3},
		shape:   []int{3},
		strides: []int{1},
		offset:  1,
	}
	if v.isContiguous() {
		t.Fatalf("offset view reported contiguous")
	}
	r := mustArr(t, ok(v.Reshape(3)))
	eqFloats(t, r.materialize(), []float64{1, 2, 3})
}

func TestIsContiguousOversizedData(t *testing.T) {
	// Correct row-major strides but a data slice longer than the shape implies:
	// not contiguous (exercises the Size() == len(data) final check returning
	// false).
	v := &Array{
		data:    []float64{1, 2, 3, 4, 5},
		shape:   []int{3},
		strides: []int{1},
	}
	if v.isContiguous() {
		t.Fatalf("oversized-data view reported contiguous")
	}
}

func TestDivByZeroProducesInf(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1}, 1)))
	z := mustArr(t, ok(FromData([]float64{0}, 1)))
	r := mustArr(t, ok(a.Div(z)))
	if !math.IsInf(r.At(0), 1) {
		t.Fatalf("1/0 = %v, want +Inf", r.At(0))
	}
}
