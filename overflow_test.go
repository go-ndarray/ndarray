package ndarray

import (
	"errors"
	"math"
	"math/bits"
	"testing"
)

// Shape extents chosen relative to the word size, so the same products
// overflow int on 32- and 64-bit targets alike (a literal 1<<62 does not even
// compile where int is 32 bits).
const (
	quarterWord = 1 << (bits.UintSize - 2) // x4 wraps to exactly 0
	byteBreaker = 1 << (bits.UintSize - 5) // x8 elements fit; x8 bytes do not
	halfSquare  = 1 << (bits.UintSize/2 + 4)
)

// TestShapeSizeOverflow checks that a shape whose element count overflows int
// is rejected instead of wrapping. Before the fix prod() wrapped silently (on
// 64-bit): New(1<<62, 4) returned an array of Size 0, and FromData/Reshape
// accepted (1<<62+1, 4) for 4 elements because the product wrapped to 4, after
// which At(1, 0) panicked. numpy raises "array is too big" for all of these.
func TestShapeSizeOverflow(t *testing.T) {
	huge := [][]int{
		{quarterWord, 4},
		{quarterWord + 1, 4},
		{byteBreaker, 8},
		{math.MaxInt, 2},
		{0, quarterWord, quarterWord}, // numpy rejects this too, despite the zero
	}
	for _, s := range huge {
		if _, err := New(s...); !errors.Is(err, ErrShapeMismatch) {
			t.Errorf("New(%v) err = %v, want ErrShapeMismatch", s, err)
		}
	}
	if _, err := FromData([]float64{1, 2, 3, 4}, quarterWord+1, 4); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("FromData wrapped shape err = %v", err)
	}
	a := mustArr(t, ok(Arange(0, 4, 1)))
	if _, err := a.Reshape(quarterWord+1, 4); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Reshape wrapped shape err = %v", err)
	}
	if _, err := a.Reshape(-1, quarterWord+1, 4); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Reshape wrapped shape with -1 err = %v", err)
	}
	// Two valid shapes can broadcast to an unrepresentable one.
	col := &Array{data: []float64{1}, shape: []int{halfSquare, 1}, strides: []int{0, 0}}
	row := &Array{data: []float64{1}, shape: []int{1, halfSquare}, strides: []int{0, 0}}
	if _, err := col.Add(row); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("broadcast to (%d, %d) err = %v", halfSquare, halfSquare, err)
	}
	// Large but representable shapes are still fine.
	if _, err := New(0, halfSquare); err != nil {
		t.Errorf("New(0, %d) err = %v", halfSquare, err)
	}
}

// TestSliceHugeStep checks the element count of a slice whose |step| exceeds
// the axis length. Before the fix ceil((stop-start)/step) was computed as
// (stop-start+step-1)/step, which overflows for a step near MaxInt and gave an
// empty view; numpy gives one element (Python a[0:5:sys.maxsize] == a[0:1]).
func TestSliceHugeStep(t *testing.T) {
	x := mustArr(t, ok(Arange(0, 6, 1)))
	for _, c := range []struct {
		name string
		ix   Index
		want []float64
	}{
		{"0:5:MaxInt", Rng(0, 5, math.MaxInt), []float64{0}},
		{"4:0:-MaxInt", Rng(4, 0, -math.MaxInt), []float64{4}},
		{"::-MaxInt", Step(-math.MaxInt), []float64{5}},
		{"::MaxInt", Step(math.MaxInt), []float64{0}},
		{"::MinInt", Step(math.MinInt), []float64{5}},
		{"5:0:MaxInt", Rng(5, 0, math.MaxInt), []float64{}},
	} {
		s, err := x.Slice(c.ix)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		wantData(t, s, []int{len(c.want)}, c.want)
	}
}

// TestZeroDimViewBroadcast checks elementwise ops on a 0-d view that does not
// start at offset 0 (x[2] of a 1-D x). Such a view is not "contiguous", so it
// takes the broadcasting path, which indexed shape[n-1] with n == 0 and
// panicked with "index out of range [-1]".
func TestZeroDimViewBroadcast(t *testing.T) {
	x := mustArr(t, ok(Arange(0, 6, 1)))
	s := mustArr(t, ok(x.Slice(A(2))))
	r := mustArr(t, ok(s.Add(s)))
	wantData(t, r, nil, []float64{4}) // Shape() of a 0-d array is nil
	r = mustArr(t, ok(s.Mul(x)))
	wantData(t, r, []int{6}, []float64{0, 2, 4, 6, 8, 10})
	m := mustArr(t, ok(s.MaskSelect(s)))
	wantData(t, m, []int{1}, []float64{2})
	w := mustArr(t, ok(Where(s, s, x)))
	wantData(t, w, []int{6}, []float64{2, 2, 2, 2, 2, 2})
}
