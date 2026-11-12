package random

import (
	"testing"

	"github.com/go-ndarray/ndarray"
)

// TestChoiceEveryDTypeMiddleAxis pins the values of the byte gather for every
// element size, with outer > 1 and inner > 1 at once (the middle axis of a
// (2, 7, 3) array). The int64 result must be a real gather: element (o, j, i)
// is o·21 + k_j·3 + i for one k_j per j. And the draws do not depend on the
// dtype, so for every dtype Choice(x.AsType(dt)) must equal
// Choice(x).AsType(dt) for the same seed: a slab of the wrong byte size would
// gather the wrong elements.
func TestChoiceEveryDTypeMiddleAxis(t *testing.T) {
	x, err := ndarray.Arange(0, 42, 1)
	if err != nil {
		t.Fatal(err)
	}
	if x, err = x.AsType(ndarray.Int64).Reshape(2, 7, 3); err != nil {
		t.Fatal(err)
	}
	const size = 5
	ref, err := DefaultRNG(7).Choice(x, ChoiceOptions{Axis: 1}, size)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := ndarray.Data[int64](ref)
	for j := range size {
		k := (got[j*3] - 0) / 3 // from (o=0, j, i=0)
		for o := range 2 {
			for i := range 3 {
				if want := int64(o*21) + k*3 + int64(i); got[(o*size+j)*3+i] != want {
					t.Fatalf("int64 [%d %d %d] = %d, want %d (index %d)", o, j, i, got[(o*size+j)*3+i], want, k)
				}
			}
		}
	}
	for _, dt := range []ndarray.DType{
		ndarray.Bool, ndarray.Int8, ndarray.Int16, ndarray.Int32, ndarray.Int64,
		ndarray.Uint8, ndarray.Uint16, ndarray.Uint32, ndarray.Uint64,
		ndarray.Float32, ndarray.Float64, ndarray.Complex64, ndarray.Complex128,
	} {
		r, err := DefaultRNG(7).Choice(x.AsType(dt), ChoiceOptions{Axis: 1}, size)
		if err != nil {
			t.Fatal(err)
		}
		if r.DType() != dt {
			t.Fatalf("%v: result dtype %v", dt, r.DType())
		}
		a, _ := ndarray.Data[complex128](r.AsType(ndarray.Complex128))
		b, _ := ndarray.Data[complex128](ref.AsType(dt).AsType(ndarray.Complex128))
		for i := range b {
			if a[i] != b[i] {
				t.Fatalf("%v: element %d = %v, want %v", dt, i, a[i], b[i])
			}
		}
	}
}

// TestChoiceAllocsConstant: the gather of Choice allocates nothing per drawn
// element (#103: it was two per element through reflect.Value.Slice), so 100
// and 10 000 draws cost the same number of allocations, along axis 0 of a 1-D
// array and along an inner axis of a 2-D one.
func TestChoiceAllocsConstant(t *testing.T) {
	a, err := ndarray.Arange(0, 1000, 1)
	if err != nil {
		t.Fatal(err)
	}
	m, err := a.Reshape(10, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		a    *ndarray.Array
		axis int
	}{{"1-D", a, 0}, {"2-D axis 1", m, 1}} {
		allocs := func(size int) float64 {
			g := DefaultRNG(1)
			return testing.AllocsPerRun(20, func() {
				if _, err := g.Choice(c.a, ChoiceOptions{Axis: c.axis}, size); err != nil {
					t.Fatal(err)
				}
			})
		}
		if small, large := allocs(100), allocs(10000); large != small {
			t.Errorf("%s: %v allocations for 100 draws, %v for 10000: the gather allocates per element", c.name, small, large)
		}
	}
}
