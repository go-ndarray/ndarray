package kernels

import (
	"math"
	"testing"
)

func eqSlice(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestArith(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	dst := make([]float64, 3)

	Add(dst, a, b)
	eqSlice(t, dst, []float64{5, 7, 9})
	Sub(dst, a, b)
	eqSlice(t, dst, []float64{-3, -3, -3})
	Mul(dst, a, b)
	eqSlice(t, dst, []float64{4, 10, 18})
	Div(dst, []float64{8, 9, 12}, []float64{2, 3, 4})
	eqSlice(t, dst, []float64{4, 3, 3})
}

func TestMap(t *testing.T) {
	src := []float64{1, -2, 3}
	dst := make([]float64, 3)
	Map(dst, src, func(x float64) float64 { return x * x })
	eqSlice(t, dst, []float64{1, 4, 9})
}

func TestReductions(t *testing.T) {
	a := []float64{3, 1, 4, 1, 5}
	if got := Sum(a); got != 14 {
		t.Errorf("Sum = %v, want 14", got)
	}
	if got := Prod([]float64{1, 2, 3, 4}); got != 24 {
		t.Errorf("Prod = %v, want 24", got)
	}
	if got := Max(a); got != 5 {
		t.Errorf("Max = %v, want 5", got)
	}
	if got := Min(a); got != 1 {
		t.Errorf("Min = %v, want 1", got)
	}
	// single-element paths for Max/Min
	if got := Max([]float64{7}); got != 7 {
		t.Errorf("Max single = %v, want 7", got)
	}
	if got := Min([]float64{7}); got != 7 {
		t.Errorf("Min single = %v, want 7", got)
	}
}

func TestAbs(t *testing.T) {
	if got := Abs(-3.5); got != 3.5 {
		t.Errorf("Abs = %v, want 3.5", got)
	}
	if got := Abs(2); got != 2 {
		t.Errorf("Abs = %v, want 2", got)
	}
	if !math.IsInf(Abs(math.Inf(-1)), 1) {
		t.Errorf("Abs(-Inf) should be +Inf")
	}
}
