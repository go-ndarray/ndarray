package fft

import (
	"fmt"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// shiftByConcat is the previous implementation of shift, kept as the oracle:
// each axis in turn, its two halves joined the other way round.
func shiftByConcat(a *ndarray.Array, axes []int, inverse bool) *ndarray.Array {
	if axes == nil {
		axes = make([]int, a.Ndim())
		for i := range axes {
			axes[i] = i
		}
	}
	var norm []int
	if len(axes) > 0 {
		norm, _, _ = resolveAxes(a, Options{Axes: axes}, allD)
	}
	x := a
	shape := a.Shape()
	for _, ax := range norm {
		n := shape[ax]
		cut := n - n/2
		if inverse {
			cut = n / 2
		}
		idx := make([]ndarray.Index, len(shape))
		for i := range idx {
			idx[i] = ndarray.All()
		}
		idx[ax] = ndarray.From(cut)
		tail, _ := x.Slice(idx...)
		idx[ax] = ndarray.To(cut)
		head, _ := x.Slice(idx...)
		x, _ = ndarray.Concatenate([]*ndarray.Array{tail, head}, ax)
	}
	return x.Copy()
}

// TestShiftMatchesConcat: the one-pass roll gives the same array (shape,
// dtype, every element) as the axis-by-axis concatenation, for 0-d to 3-D
// shapes with even, odd, unit and empty axes, axis subsets (negative and
// repeated included), several dtypes and a non-contiguous input.
func TestShiftMatchesConcat(t *testing.T) {
	shapes := [][]int{{}, {1}, {5}, {6}, {0}, {3, 4}, {4, 3}, {1, 7}, {2, 0}, {3, 4, 5}, {2, 1, 6}}
	axesSets := [][]int{nil, {0}, {-1}, {0, 0}, {-1, 0}, {1}, {2}, {0, 2}}
	for _, sh := range shapes {
		size := 1
		for _, n := range sh {
			size *= n
		}
		vals := make([]float64, size)
		for i := range vals {
			vals[i] = float64(i*7%11) - 3
		}
		f64, err := ndarray.FromSlice(vals, sh...)
		if err != nil {
			t.Fatal(err)
		}
		inputs := map[string]*ndarray.Array{
			"float64":    f64,
			"int16":      f64.AsType(ndarray.Int16),
			"complex128": f64.AsType(ndarray.Complex128),
			"bool":       f64.AsType(ndarray.Bool),
		}
		if len(sh) >= 2 {
			inputs["transposed"] = f64.Transpose() // not contiguous
		}
		for name, x := range inputs {
			for _, axes := range axesSets {
				if !axesValid(x.Ndim(), axes) {
					continue
				}
				for _, inverse := range []bool{false, true} {
					got, err := shift(x, axes, inverse)
					if err != nil {
						t.Fatalf("%s %v axes=%v inverse=%v: %v", name, sh, axes, inverse, err)
					}
					want := shiftByConcat(x, axes, inverse)
					if got.DType() != want.DType() || fmt.Sprint(got.Shape()) != fmt.Sprint(want.Shape()) {
						t.Fatalf("%s %v axes=%v inverse=%v: %v %v, want %v %v", name, sh, axes, inverse, got.DType(), got.Shape(), want.DType(), want.Shape())
					}
					if g, w := got.Repr(), want.Repr(); g != w {
						t.Fatalf("%s %v axes=%v inverse=%v:\n got %s\nwant %s", name, sh, axes, inverse, g, w)
					}
				}
			}
		}
	}
}

var shiftSink *ndarray.Array

// The table's case (#118): a 256×256 float64 array, every axis.
func BenchmarkFFTShift256(b *testing.B) {
	vals := make([]float64, 256*256)
	for i := range vals {
		vals[i] = float64(i)
	}
	x, _ := ndarray.FromSlice(vals, 256, 256)
	for b.Loop() {
		shiftSink, _ = FFTShift(x)
	}
}

// axesValid reports whether every axis in axes exists for an ndim-D array.
func axesValid(ndim int, axes []int) bool {
	for _, a := range axes {
		if a < -ndim || a >= ndim {
			return false
		}
	}
	return true
}

// TestShiftDoesNotAliasItsInput: the result is a new array even when nothing
// moves (no axis, or a length-1 axis), as before.
func TestShiftDoesNotAliasItsInput(t *testing.T) {
	x, _ := ndarray.FromSlice([]float64{1, 2, 3}, 3)
	r, _ := FFTShift(x, []int{}...)
	r.Set(99, 0)
	if x.At(0) != 1 {
		t.Fatalf("FFTShift with no axis aliases its input: %v", x.At(0))
	}
}
