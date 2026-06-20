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
	if _, err := a.Reshape(-2); !errors.Is(err, ErrShapeMismatch) {
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

func TestBinOpInto(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	b := mustArr(t, ok(FromData([]float64{5, 6, 7, 8}, 2, 2)))
	out := mustArr(t, ok(New(2, 2)))

	if err := a.AddInto(out, b); err != nil {
		t.Fatalf("AddInto: %v", err)
	}
	eqFloats(t, out.materialize(), []float64{6, 8, 10, 12})
	if err := a.SubInto(out, b); err != nil {
		t.Fatalf("SubInto: %v", err)
	}
	eqFloats(t, out.materialize(), []float64{-4, -4, -4, -4})
	if err := a.MulInto(out, b); err != nil {
		t.Fatalf("MulInto: %v", err)
	}
	eqFloats(t, out.materialize(), []float64{5, 12, 21, 32})
	if err := b.DivInto(out, a); err != nil {
		t.Fatalf("DivInto: %v", err)
	}
	eqFloats(t, out.materialize(), []float64{5, 3, 7.0 / 3.0, 2})

	// Aliasing: out == a is allowed (each index read before written).
	c := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	if err := c.AddInto(c, b); err != nil {
		t.Fatalf("AddInto alias: %v", err)
	}
	eqFloats(t, c.materialize(), []float64{6, 8, 10, 12})

	// Broadcasting operands into a full-shape out.
	row := mustArr(t, ok(FromData([]float64{10, 20}, 2)))
	o2 := mustArr(t, ok(New(2, 2)))
	if err := a.AddInto(o2, row); err != nil {
		t.Fatalf("AddInto broadcast: %v", err)
	}
	eqFloats(t, o2.materialize(), []float64{11, 22, 13, 24})
}

func TestBinOpIntoErrors(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	b := mustArr(t, ok(FromData([]float64{5, 6, 7, 8}, 2, 2)))

	// Wrong out shape.
	small := mustArr(t, ok(New(2)))
	if err := a.AddInto(small, b); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected shape error, got %v", err)
	}
	// Non-contiguous out (a transposed view).
	view := mustArr(t, ok(New(2, 2))).Transpose()
	if err := a.AddInto(view, b); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected contiguity error, got %v", err)
	}
	// Incompatible operand shapes.
	bad := mustArr(t, ok(FromData([]float64{1, 2, 3}, 3)))
	out := mustArr(t, ok(New(2, 2)))
	if err := a.AddInto(out, bad); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected broadcast error, got %v", err)
	}
}

func TestSqrtInto(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 4, 9, 16}, 2, 2)))
	out := mustArr(t, ok(New(2, 2)))
	if err := a.SqrtInto(out); err != nil {
		t.Fatalf("SqrtInto: %v", err)
	}
	eqFloats(t, out.materialize(), []float64{1, 2, 3, 4})
	// Alias: out == a.
	if err := a.SqrtInto(a); err != nil {
		t.Fatalf("SqrtInto alias: %v", err)
	}
	eqFloats(t, a.materialize(), []float64{1, 2, 3, 4})
	// Errors: wrong shape and non-contiguous out.
	if err := a.SqrtInto(mustArr(t, ok(New(2)))); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected shape error, got %v", err)
	}
	view := mustArr(t, ok(New(2, 2))).Transpose()
	if err := a.SqrtInto(view); !errors.Is(err, ErrBroadcast) {
		t.Fatalf("expected contiguity error, got %v", err)
	}
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

// TestMaxMinNaN locks the NaN convention of the public Max/Min reductions:
// they are NaN-propagating (any NaN element -> NaN result), matching numpy.max /
// numpy.min and Go's builtin max/min. This is the documented, corrected
// semantics (the earlier `if v > m` form silently ignored NaNs).
func TestMaxMinNaN(t *testing.T) {
	nan := math.NaN()
	for _, data := range [][]float64{
		{nan, 1, 2, 3},
		{1, 2, nan, 3},
		{1, 2, 3, nan},
		{nan},
	} {
		a := mustArr(t, ok(FromData(data, len(data))))
		mx, err := a.Max()
		if err != nil || !math.IsNaN(mx) {
			t.Fatalf("Max(%v) = %v, err %v; want NaN", data, mx, err)
		}
		mn, err := a.Min()
		if err != nil || !math.IsNaN(mn) {
			t.Fatalf("Min(%v) = %v, err %v; want NaN", data, mn, err)
		}
	}
	// Without NaN the extremes are exact, including +/-Inf.
	a := mustArr(t, ok(FromData([]float64{math.Inf(-1), -5, 0, 5, math.Inf(1)}, 5)))
	if mx, _ := a.Max(); mx != math.Inf(1) {
		t.Fatalf("Max with +Inf = %v", mx)
	}
	if mn, _ := a.Min(); mn != math.Inf(-1) {
		t.Fatalf("Min with -Inf = %v", mn)
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

func TestAxisReductions2D(t *testing.T) {
	// b = [[1,2,3],[4,5,6]]; expectations from numpy 2.4.6.
	b := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))

	s0 := mustArr(t, ok(b.SumAxis(0, false)))
	eqInts(t, s0.Shape(), []int{3})
	eqFloats(t, s0.materialize(), []float64{5, 7, 9})

	s1 := mustArr(t, ok(b.SumAxis(1, false)))
	eqInts(t, s1.Shape(), []int{2})
	eqFloats(t, s1.materialize(), []float64{6, 15})

	// negative axis: -1 == last axis.
	sNeg := mustArr(t, ok(b.SumAxis(-1, false)))
	eqFloats(t, sNeg.materialize(), []float64{6, 15})

	p1 := mustArr(t, ok(b.ProdAxis(1, false)))
	eqFloats(t, p1.materialize(), []float64{6, 120})

	mx1 := mustArr(t, ok(b.MaxAxis(1, false)))
	eqFloats(t, mx1.materialize(), []float64{3, 6})

	mn0 := mustArr(t, ok(b.MinAxis(0, false)))
	eqFloats(t, mn0.materialize(), []float64{1, 2, 3})

	mean0 := mustArr(t, ok(b.MeanAxis(0, false)))
	eqFloats(t, mean0.materialize(), []float64{2.5, 3.5, 4.5})
	mean1 := mustArr(t, ok(b.MeanAxis(1, false)))
	eqFloats(t, mean1.materialize(), []float64{2, 5})
}

func TestAxisReductionsKeepdims(t *testing.T) {
	b := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	k := mustArr(t, ok(b.SumAxis(0, true)))
	eqInts(t, k.Shape(), []int{1, 3})
	eqFloats(t, k.materialize(), []float64{5, 7, 9})

	km := mustArr(t, ok(b.MeanAxis(1, true)))
	eqInts(t, km.Shape(), []int{2, 1})
	eqFloats(t, km.materialize(), []float64{2, 5})
}

func TestAxisReductions3D(t *testing.T) {
	// a = arange(24).reshape(2,3,4); expectations from numpy 2.4.6.
	data := make([]float64, 24)
	for i := range data {
		data[i] = float64(i)
	}
	a := mustArr(t, ok(FromData(data, 2, 3, 4)))

	s0 := mustArr(t, ok(a.SumAxis(0, false)))
	eqInts(t, s0.Shape(), []int{3, 4})
	eqFloats(t, s0.materialize(),
		[]float64{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})

	s1 := mustArr(t, ok(a.SumAxis(1, false)))
	eqInts(t, s1.Shape(), []int{2, 4})
	eqFloats(t, s1.materialize(),
		[]float64{12, 15, 18, 21, 48, 51, 54, 57})

	s2 := mustArr(t, ok(a.SumAxis(2, false)))
	eqInts(t, s2.Shape(), []int{2, 3})
	eqFloats(t, s2.materialize(), []float64{6, 22, 38, 54, 70, 86})

	// negative axis on 3-D, with keepdims shape (2,3,1).
	sNeg := mustArr(t, ok(a.SumAxis(-1, true)))
	eqInts(t, sNeg.Shape(), []int{2, 3, 1})
	eqFloats(t, sNeg.materialize(), []float64{6, 22, 38, 54, 70, 86})

	mx1 := mustArr(t, ok(a.MaxAxis(1, false)))
	eqFloats(t, mx1.materialize(),
		[]float64{8, 9, 10, 11, 20, 21, 22, 23})
	mn1 := mustArr(t, ok(a.MinAxis(1, false)))
	eqFloats(t, mn1.materialize(),
		[]float64{0, 1, 2, 3, 12, 13, 14, 15})
	p1 := mustArr(t, ok(a.ProdAxis(1, false)))
	eqFloats(t, p1.materialize(),
		[]float64{0, 45, 120, 231, 3840, 4641, 5544, 6555})
	mean2 := mustArr(t, ok(a.MeanAxis(2, false)))
	eqFloats(t, mean2.materialize(),
		[]float64{1.5, 5.5, 9.5, 13.5, 17.5, 21.5})
}

func TestAxisReduction1D(t *testing.T) {
	// Reducing a 1-D array over axis 0 yields a 0-d (scalar) array.
	c := mustArr(t, ok(FromData([]float64{3, 1, 4, 1, 5}, 5)))
	s := mustArr(t, ok(c.SumAxis(0, false)))
	if s.Ndim() != 0 || s.Size() != 1 {
		t.Fatalf("1-D reduce: Ndim=%d Size=%d", s.Ndim(), s.Size())
	}
	eqFloats(t, s.materialize(), []float64{14})

	k := mustArr(t, ok(c.SumAxis(0, true)))
	eqInts(t, k.Shape(), []int{1})
	eqFloats(t, k.materialize(), []float64{14})
}

func TestAxisReductionStridedInput(t *testing.T) {
	// A transposed (non-contiguous) view must reduce correctly: materialize is
	// exercised inside reduceAxis. tr of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)))
	tr := a.Transpose()
	eqInts(t, tr.Shape(), []int{3, 2})
	s := mustArr(t, ok(tr.SumAxis(1, false)))
	eqInts(t, s.Shape(), []int{3})
	eqFloats(t, s.materialize(), []float64{5, 7, 9})
}

func TestAxisReductionErrors(t *testing.T) {
	a := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	if _, err := a.SumAxis(2, false); !errors.Is(err, ErrAxis) {
		t.Fatalf("axis too high: %v", err)
	}
	if _, err := a.SumAxis(-3, false); !errors.Is(err, ErrAxis) {
		t.Fatalf("axis too low: %v", err)
	}
	// Every public reduction shares normalizeAxis; check they all surface ErrAxis.
	for name, fn := range map[string]func(int, bool) (*Array, error){
		"Prod": a.ProdAxis, "Max": a.MaxAxis, "Min": a.MinAxis, "Mean": a.MeanAxis,
	} {
		if _, err := fn(9, false); !errors.Is(err, ErrAxis) {
			t.Fatalf("%sAxis bad axis: %v", name, err)
		}
	}

	// Reduction along a zero-length axis is rejected (empty-reduction case),
	// for both the kernel-backed reductions and the Sum-derived Mean.
	z := mustArr(t, ok(New(0, 3)))
	if _, err := z.SumAxis(0, false); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("zero-axis Sum: %v", err)
	}
	if _, err := z.MeanAxis(0, false); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("zero-axis Mean: %v", err)
	}
	// Reducing a non-zero axis of an array that still has zero elements keeps an
	// empty result rather than erroring (axisLen > 0 but inner produces nothing).
	zr, err := z.SumAxis(1, false)
	if err != nil {
		t.Fatalf("axis-1 reduce of (0,3): %v", err)
	}
	eqInts(t, zr.Shape(), []int{0})
	if zr.Size() != 0 {
		t.Fatalf("expected empty result, size %d", zr.Size())
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
