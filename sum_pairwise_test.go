package ndarray

import (
	"math"
	"math/big"
	"testing"
)

// Sums along a contiguous axis of float32 and complex arrays are pairwise, as
// NumPy's are (FLOAT/CFLOAT_pairwise_sum, numpy/_core/src/umath/
// loops_utils.h.src). The expectations below come from exact arithmetic, not
// from running the code.

// TestFloat32SumDoesNotSaturate: 2^24 followed by 1000 ones. Added in order
// in float32, every 2^24 + 1 rounds back to 2^24, so the sum stays 2^24.
// Pairwise, the ones are summed among themselves first (1000, exact), and
// 2^24 + 1000 is representable (float32's spacing at 2^24 is 2).
func TestFloat32SumDoesNotSaturate(t *testing.T) {
	const n = 1000
	v := make([]float32, n+1)
	v[0] = 1 << 24
	for i := 1; i <= n; i++ {
		v[i] = 1
	}
	a := must(FromSlice(v, len(v)))
	if got, want := a.Sum(), float64(1<<24+n); got != want {
		t.Errorf("Sum = %v, want %v (in order it would be %v)", got, want, float64(1<<24))
	}
	// The same along the last axis of a 2-D array, row by row.
	two := must(FromSlice(append(append([]float32{}, v...), v...), 2, len(v)))
	rows, ok := Data[float32](must(two.SumAxis(1, false)))
	if !ok || len(rows) != 2 {
		t.Fatalf("SumAxis(1) of a 2-row float32 array: %v, %v", rows, ok)
	}
	for i, got := range rows {
		if got != 1<<24+n {
			t.Errorf("row %d: %v, want %v", i, got, 1<<24+n)
		}
	}
	c := make([]complex64, n+1)
	c[0] = complex(1<<24, -(1 << 24))
	for i := 1; i <= n; i++ {
		c[i] = complex(1, -1)
	}
	got := Item[complex64](must(FromSlice(c, len(c))).SumAll())
	if want := complex64(complex(1<<24+n, -(1<<24 + n))); got != want {
		t.Errorf("complex64 sum = %v, want %v", got, want)
	}
}

// TestFloat32SumError compares the sum of 2^16 float32 values of mixed sign
// and magnitude with the exact sum (math/big): pairwise must be far closer
// than a sum in order.
func TestFloat32SumError(t *testing.T) {
	const n = 1 << 16
	v := make([]float32, n)
	x := uint32(12345)
	for i := range v {
		x = x*1664525 + 1013904223 // a fixed LCG: reproducible, no seed to drift
		v[i] = float32(int32(x)>>8) / float32(1<<8) * float32(1+int(x%7))
	}
	exact := new(big.Float).SetPrec(256)
	var inOrder float32
	for _, f := range v {
		exact.Add(exact, new(big.Float).SetFloat64(float64(f)))
		inOrder += f
	}
	e, _ := exact.Float64()
	got := must(FromSlice(v, n)).Sum()
	errPairwise, errInOrder := math.Abs(got-e), math.Abs(float64(inOrder)-e)
	t.Logf("exact %v: pairwise %v (error %g), in order %v (error %g)", e, got, errPairwise, inOrder, errInOrder)
	if errPairwise*10 > errInOrder {
		t.Errorf("pairwise error %g is not ten times below the in-order error %g", errPairwise, errInOrder)
	}
}

// TestProdAlongANonContiguousAxis: a product along axis 0 of a 2-D integer
// array keeps the row-by-row loop (the reduced axis is not contiguous), and
// accumulates in int64 as NumPy does for int32.
func TestProdAlongANonContiguousAxis(t *testing.T) {
	a := must(FromSlice([]int32{1, 2, 3, 4, 5, -6}, 2, 3))
	p := must(a.ProdAxis(0, false))
	got, ok := Data[int64](p)
	if !ok || len(got) != 3 || got[0] != 4 || got[1] != 10 || got[2] != -18 {
		t.Errorf("ProdAxis(0) = %v %v, want int64 [4 10 -18]", p.DType(), got)
	}
}

// TestWidenedSums: sums and products of bools and narrow integers along a
// contiguous axis fold each element into the 64-bit accumulator as it is
// read. The expectation widens each element first and folds with Go's
// arithmetic of the accumulator, wrap-around included; a transposed operand
// takes the gather path.
func TestWidenedSums(t *testing.T) {
	const rows, cols = 3, 37
	check := func(name string, a *Array, wantSum, wantProd func(row []int) any) {
		t.Helper()
		for _, x := range []*Array{a, must(a.Transpose().Copy().Reshape(cols, rows)).Transpose()} {
			s, p := must(x.SumAxis(1, false)), must(x.ProdAxis(1, false))
			for r := range rows {
				idx := []int{r}
				if gs, ws := s.itemAny(idx), wantSum([]int{r}); gs != ws {
					t.Errorf("%s sum row %d = %v, want %v", name, r, gs, ws)
				}
				if gp, wp := p.itemAny(idx), wantProd([]int{r}); gp != wp {
					t.Errorf("%s prod row %d = %v, want %v", name, r, gp, wp)
				}
			}
		}
	}
	i32 := make([]int32, rows*cols)
	u16 := make([]uint16, rows*cols)
	b := make([]bool, rows*cols)
	for i := range i32 {
		i32[i] = int32(math.MaxInt32 - i*7919) // sums past 2^32, products that wrap
		u16[i] = uint16(65535 - i*31)
		b[i] = i%11 != 0
	}
	b[cols+5] = true // row 1 all true except its i%11 == 0 slots
	check("int32", must(FromSlice(i32, rows, cols)), func(r []int) any {
		var acc int64
		for _, v := range i32[r[0]*cols : (r[0]+1)*cols] {
			acc += int64(v)
		}
		return acc
	}, func(r []int) any {
		var acc int64 = 1
		for _, v := range i32[r[0]*cols : (r[0]+1)*cols] {
			acc *= int64(v)
		}
		return acc
	})
	check("uint16", must(FromSlice(u16, rows, cols)), func(r []int) any {
		var acc uint64
		for _, v := range u16[r[0]*cols : (r[0]+1)*cols] {
			acc += uint64(v)
		}
		return acc
	}, func(r []int) any {
		var acc uint64 = 1
		for _, v := range u16[r[0]*cols : (r[0]+1)*cols] {
			acc *= uint64(v)
		}
		return acc
	})
	check("bool", must(FromSlice(b, rows, cols)), func(r []int) any {
		var acc int64
		for _, v := range b[r[0]*cols : (r[0]+1)*cols] {
			if v {
				acc++
			}
		}
		return acc
	}, func(r []int) any {
		var acc int64 = 1
		for _, v := range b[r[0]*cols : (r[0]+1)*cols] {
			if !v {
				acc = 0
			}
		}
		return acc
	})
	for _, dt := range []DType{Int8, Int16, Uint8, Uint32} {
		x := must(Arange(1, 10, 1)).AsType(dt)
		if got := x.SumAll().itemAny([]int{}); got != map[bool]any{true: int64(45), false: uint64(45)}[dt.IsSigned()] {
			t.Errorf("%v: SumAll = %v, want 45 in the accumulator type", dt, got)
		}
		if got := x.ProdAll().itemAny([]int{}); got != map[bool]any{true: int64(362880), false: uint64(362880)}[dt.IsSigned()] {
			t.Errorf("%v: ProdAll = %v, want 362880", dt, got)
		}
	}
}

// itemAny returns element idx of a in its Go type, for comparisons.
func (a *Array) itemAny(idx []int) any {
	s := a.contiguousStore()
	f := flatIndex(idx, a.shape)
	switch st := s.(type) {
	case []int64:
		return st[f]
	case []uint64:
		return st[f]
	}
	return nil
}

// TestPairwiseSumBranches covers each length class of NumPy's scheme and
// checks it against the exact sum of small integers (exact in float32), and
// that a sum of only -0 stays -0.
func TestPairwiseSumBranches(t *testing.T) {
	for _, n := range []int{1, 2, 7, 8, 9, 15, 16, 127, 128, 129, 130, 1000, 4099} {
		v := make([]float32, n)
		c := make([]complex64, n)
		want := 0
		for i := range v {
			v[i] = float32(i%13 - 6)
			c[i] = complex(float32(i%5), float32(-(i % 3)))
			want += i%13 - 6
		}
		if got := must(FromSlice(v, n)).Sum(); got != float64(want) {
			t.Errorf("n=%d: float32 sum %v, want %d", n, got, want)
		}
		var wre, wim int
		for i := range c {
			wre += i % 5
			wim -= i % 3
		}
		if got := Item[complex64](must(FromSlice(c, n)).SumAll()); got != complex(float32(wre), float32(wim)) {
			t.Errorf("n=%d: complex64 sum %v, want (%d%+di)", n, got, wre, wim)
		}
	}
	negZero := float32(math.Copysign(0, -1))
	if got := must(FromSlice([]float32{negZero, negZero, negZero}, 3)).Sum(); !math.Signbit(got) {
		t.Errorf("a sum of -0 is %v, want -0", got)
	}
	nz := complex(negZero, negZero)
	if got := Item[complex64](must(FromSlice([]complex64{nz, nz}, 2)).SumAll()); !math.Signbit(float64(real(got))) || !math.Signbit(float64(imag(got))) {
		t.Errorf("a complex sum of -0 is %v, want -0-0i", got)
	}
}
