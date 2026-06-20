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

func TestAxisKernels(t *testing.T) {
	// src laid out as [outer=2][axisLen=3][inner=2]:
	// block0 rows {1,2},{3,4},{5,6}; block1 rows {7,8},{9,10},{11,12}.
	src := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	dst := make([]float64, 2*2)

	SumAxis(dst, src, 2, 3, 2)
	eqSlice(t, dst, []float64{9, 12, 27, 30})

	ProdAxis(dst, src, 2, 3, 2)
	eqSlice(t, dst, []float64{15, 48, 693, 960})

	MaxAxis(dst, src, 2, 3, 2)
	eqSlice(t, dst, []float64{5, 6, 11, 12})

	MinAxis(dst, src, 2, 3, 2)
	eqSlice(t, dst, []float64{1, 2, 7, 8})
	// Descending data so MinAxis takes its "found a smaller value" branch.
	MinAxis(dst, []float64{6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7}, 2, 3, 2)
	eqSlice(t, dst, []float64{2, 1, 8, 7})
	// Descending data for MaxAxis exercises its non-update path too.
	MaxAxis(dst, []float64{6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7}, 2, 3, 2)
	eqSlice(t, dst, []float64{6, 5, 12, 11})

	// axisLen == 1: the reduce loop body never runs; result is the input.
	one := make([]float64, 2)
	SumAxis(one, []float64{4, 9}, 2, 1, 1)
	eqSlice(t, one, []float64{4, 9})
	ProdAxis(one, []float64{4, 9}, 2, 1, 1)
	eqSlice(t, one, []float64{4, 9})
	MaxAxis(one, []float64{4, 9}, 2, 1, 1)
	eqSlice(t, one, []float64{4, 9})
	MinAxis(one, []float64{4, 9}, 2, 1, 1)
	eqSlice(t, one, []float64{4, 9})
}

func BenchmarkSumAxis(b *testing.B) {
	const outer, axisLen, inner = 64, 64, 64
	src := make([]float64, outer*axisLen*inner)
	for i := range src {
		src[i] = float64(i % 7)
	}
	dst := make([]float64, outer*inner)
	b.ReportAllocs()
	b.SetBytes(int64(len(src) * 8))
	for i := 0; i < b.N; i++ {
		SumAxis(dst, src, outer, axisLen, inner)
	}
}

func TestArgAndScan(t *testing.T) {
	// ArgMax/ArgMin with the extreme appearing AFTER index 0 (exercises the
	// update branch) and ties resolving to the lowest index.
	a := []float64{3, 1, 9, 9, 0, 0}
	if got := ArgMax(a); got != 2 {
		t.Errorf("ArgMax = %d, want 2", got)
	}
	if got := ArgMin(a); got != 4 {
		t.Errorf("ArgMin = %d, want 4", got)
	}
	// Monotonic ascending/descending so neither update branch is skipped.
	if got := ArgMax([]float64{1, 2, 3}); got != 2 {
		t.Errorf("ArgMax asc = %d, want 2", got)
	}
	if got := ArgMin([]float64{3, 2, 1}); got != 2 {
		t.Errorf("ArgMin desc = %d, want 2", got)
	}

	// Axis variants over [outer=1][axisLen=3][inner=2]: columns {3,1},{9,5} etc.
	src := []float64{3, 1, 9, 5, 0, 7}
	dst := make([]float64, 2)
	ArgMaxAxis(dst, src, 1, 3, 2)
	eqSlice(t, dst, []float64{1, 2}) // col0 max at k=1 (9); col1 max at k=2 (7)
	ArgMinAxis(dst, src, 1, 3, 2)
	eqSlice(t, dst, []float64{2, 0}) // col0 min at k=2 (0); col1 min at k=0 (1)

	cs := make([]float64, 6)
	CumSumAxis(cs, src, 1, 3, 2)
	eqSlice(t, cs, []float64{3, 1, 12, 6, 12, 13})
	cp := make([]float64, 6)
	CumProdAxis(cp, src, 1, 3, 2)
	eqSlice(t, cp, []float64{3, 1, 27, 5, 0, 35})
}

func TestClipAndWhere(t *testing.T) {
	dst := make([]float64, 5)
	Clip(dst, []float64{-2, 0, 3, 7, 10}, 0, 6)
	eqSlice(t, dst, []float64{0, 0, 3, 6, 6})

	Where(dst, []float64{1, 0, 1, 0, 2}, []float64{10, 11, 12, 13, 14}, []float64{-1, -2, -3, -4, -5})
	eqSlice(t, dst, []float64{10, -2, 12, -4, 14})
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
