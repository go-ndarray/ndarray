package ndarray

import (
	"errors"
	"strconv"
	"testing"
	"time"
)

// hugeAxis is a quarter of the largest size: 2^58 on 64-bit, 2^26 on 32-bit.
const hugeAxis = (maxSize + 1) / 4

// A result whose shape cannot be represented is an error, as it is for New,
// even when no operand holds any data: zero-size operands bound nothing.
// Before, these wrapped: Concatenate produced a negative dimension, MatMul a
// shape whose Size() was 0, and the rest panicked in makeslice.
func TestResultShapeOverflow(t *testing.T) {
	must := func(a *Array, err error) *Array {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	// Sizes derive from the word size, so the test compiles and means the
	// same on 32-bit targets: big*big overflows maxSize, slab*4 exceeds it.
	big := 1 << (strconv.IntSize / 2)
	wide, tall := must(New(big, 0)), must(New(0, big))
	slab := must(New(hugeAxis, 0))
	many := func(n int) []*Array {
		s := make([]*Array, n)
		for i := range s {
			s[i] = slab
		}
		return s
	}
	for name, f := range map[string]func() (*Array, error){
		"MatMul (2^32,0)@(0,2^32)": func() (*Array, error) { return wide.MatMul(tall) },
		"MatMul m*n wraps to -1":   func() (*Array, error) { return must(New(big+1, 0)).MatMul(must(New(0, big-1))) },
		"Dot 2-D":                  func() (*Array, error) { return wide.Dot(tall) },
		"Linspace 2^62":            func() (*Array, error) { return Linspace(0, 1, maxSize+1) },
		"Concatenate sum wraps":    func() (*Array, error) { return Concatenate(many(33), 0) },
		"Stack 9 of (2^58,0)":      func() (*Array, error) { return Stack(many(9), 0) },
	} {
		a, err := f()
		if !errors.Is(err, ErrShapeMismatch) {
			t.Errorf("%s: got %v, %v; want ErrShapeMismatch", name, a, err)
		}
	}
	// Up to the limit (maxSize = 2^60-1 elements, as for New) is accepted:
	// 3 x 2^58 fits, 4 x 2^58 = 2^60 does not.
	if _, err := Concatenate(many(3), 0); err != nil {
		t.Errorf("Concatenate 3 of (2^58,0): %v", err)
	}
	if _, err := Concatenate(many(4), 0); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Concatenate 4 of (2^58,0) = (2^60,0): got %v, want ErrShapeMismatch", err)
	}
}

// An empty array can have a huge axis, (2^58, 0), and an operation that walks
// that axis without doing anything still takes years. Concatenate, the axis
// reductions and the scans did; each must now return at once.
func TestEmptyArraysWithHugeAxesReturnPromptly(t *testing.T) {
	const H = hugeAxis
	for _, sh := range [][]int{{H, 0}, {0, H}, {H, 0, 3}, {3, H, 0}, {H, 1, 0}} {
		a, err := New(sh...)
		if err != nil {
			t.Fatal(err)
		}
		ops := []func() (*Array, error){
			func() (*Array, error) { return Concatenate([]*Array{a, a}, -1) },
			func() (*Array, error) { return Concatenate([]*Array{a, a}, 0) },
			func() (*Array, error) {
				if sh[0] == 0 {
					// Summing the empty axis of (0, H) is H zeros, as in numpy: a
					// real allocation of H elements, not a walk over nothing.
					return nil, nil
				}
				return a.SumAxis(0, false)
			},
			func() (*Array, error) { return a.MaxAxis(0, true) },
			func() (*Array, error) { return a.MeanAxis(0, false) },
			func() (*Array, error) { return a.ArgMaxAxis(0, false) },
			func() (*Array, error) { return a.CumSum(0) },
			func() (*Array, error) { return a.CumProd(-1) },
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			for _, op := range ops {
				if r, err := op(); err == nil && r != nil && r.Size() != 0 {
					t.Errorf("%v: a result with %d elements from an empty array", sh, r.Size())
				}
			}
		}()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			t.Fatalf("%v: an operation on an empty array is walking its huge axis", sh)
		}
	}
}
