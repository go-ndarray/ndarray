package ndarray

import "testing"

// TestRavelIsAViewWhenContiguous: as numpy.ravel, Ravel shares the storage of
// a contiguous array (a Set through it shows in the original) and copies a
// non-contiguous one (a Set through it does not).
func TestRavelIsAViewWhenContiguous(t *testing.T) {
	a, _ := Arange(0, 6, 1)
	m, _ := a.Reshape(2, 3)

	r := m.Ravel()
	if got := r.Shape(); len(got) != 1 || got[0] != 6 {
		t.Fatalf("shape %v, want [6]", got)
	}
	r.Set(99, 4)
	if got := m.At(1, 1); got != 99 {
		t.Fatalf("contiguous Ravel is not a view: m[1,1] = %v, want 99", got)
	}

	tr := m.Transpose().Ravel() // non-contiguous: a copy, in row-major order of the transpose
	want := []float64{0, 3, 1, 99, 2, 5}
	for i, w := range want {
		if got := tr.At(i); got != w {
			t.Fatalf("transposed Ravel[%d] = %v, want %v", i, got, w)
		}
	}
	tr.Set(-1, 0)
	if got := m.At(0, 0); got != 0 {
		t.Fatalf("non-contiguous Ravel aliases the original: m[0,0] = %v", got)
	}

	// Another dtype goes through the same path.
	i8, _ := FromSlice([]int8{1, 2, 3, 4}, 2, 2)
	ri := i8.Ravel()
	ri.SetComplex(7, 3)
	if got := Item[int8](i8, 1, 1); got != 7 {
		t.Fatalf("contiguous int8 Ravel is not a view: %v, want 7", got)
	}

	// Flatten keeps copying, as numpy.flatten.
	f := m.Flatten()
	f.Set(-5, 0)
	if got := m.At(0, 0); got != 0 {
		t.Fatalf("Flatten aliases the original: m[0,0] = %v", got)
	}
}

// The table's cases (#118): 10⁵ float64, contiguous.
func BenchmarkRavelContiguous(b *testing.B) {
	a, _ := Arange(0, 100_000, 1)
	m, _ := a.Reshape(100, 1000)
	for b.Loop() {
		sinkA = m.Ravel()
	}
}

func BenchmarkRealOfReal(b *testing.B) {
	a, _ := Arange(0, 100_000, 1)
	for b.Loop() {
		sinkA = a.Real()
	}
}

// TestRealIsAViewOfARealArray: as numpy.real, Real of a non-complex array is
// the array itself (a view, same layout); Real of a complex array is a new
// float array.
func TestRealIsAViewOfARealArray(t *testing.T) {
	a, _ := Arange(0, 6, 1)
	m, _ := a.Reshape(2, 3)
	tv := m.Transpose()

	r := tv.Real()
	if got := r.Shape(); got[0] != 3 || got[1] != 2 {
		t.Fatalf("shape %v, want [3 2]", got)
	}
	if got := r.At(2, 1); got != 5 {
		t.Fatalf("Real of a transposed view reads %v, want 5", got)
	}
	r.Set(42, 2, 1)
	if got := m.At(1, 2); got != 42 {
		t.Fatalf("Real of a real array is not a view: m[1,2] = %v, want 42", got)
	}

	c, _ := FromSlice([]complex128{1 + 2i, 3 + 4i}, 2)
	rc := c.Real()
	if rc.DType() != Float64 || rc.At(1) != 3 {
		t.Fatalf("Real of complex128 = %v %v, want Float64 3", rc.DType(), rc.At(1))
	}
	rc.Set(-9, 0)
	if got := real(Item[complex128](c, 0)); got != 1 {
		t.Fatalf("Real of a complex array aliases it: %v, want 1", got)
	}

	w := Scalar(2.5).Real()
	if w.weak != weakFloat {
		t.Fatalf("Real drops the weak-scalar flag")
	}
}
