package kernels

import (
	"fmt"
	"math"
	"testing"
)

// special32 are the values a lane-wise kernel must treat as the scalar loop
// does: signed zeros, infinities, NaN, subnormals and the extremes.
var special32 = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.1, -3.5,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
	math.MaxFloat32, -math.MaxFloat32, 1e-38, 3e38, 7,
}

func operands32(n, seed int) []float32 {
	a := make([]float32, n)
	for i := range a {
		a[i] = special32[(i*7+seed)%len(special32)]
		if (i+seed)%5 == 0 {
			a[i] = float32(i+seed) / 3
		}
	}
	return a
}

func sameBits32(x, y float32) bool {
	if x != x && y != y {
		return true // NaN payloads are not part of the IEEE result
	}
	return math.Float32bits(x) == math.Float32bits(y)
}

// TestBin32MatchesScalar checks every op against the scalar oracle, bit for
// bit, at every length around the 16-float block and the tail.
func TestBin32MatchesScalar(t *testing.T) {
	for op := Add32; op <= Div32; op++ {
		for n := 0; n <= 70; n++ {
			a, b := operands32(n, 1), operands32(n, 4)
			got, want := make([]float32, n), make([]float32, n)
			Bin32(op, got, a, b)
			bin32Go(op, want, a, b)
			for i := range got {
				if !sameBits32(got[i], want[i]) {
					t.Fatalf("op %d n=%d [%d]: %v OP %v = %v, scalar %v", op, n, i, a[i], b[i], got[i], want[i])
				}
			}
		}
	}
}

// TestBin32InPlace runs dst aliasing a, the way an in-place update calls it.
func TestBin32InPlace(t *testing.T) {
	for op := Add32; op <= Div32; op++ {
		a, b := operands32(53, 2), operands32(53, 3)
		want := make([]float32, len(a))
		bin32Go(op, want, a, b)
		Bin32(op, a, a, b)
		for i := range a {
			if !sameBits32(a[i], want[i]) {
				t.Fatalf("op %d [%d]: in place %v, scalar %v", op, i, a[i], want[i])
			}
		}
	}
}

// TestSqrt32MatchesScalar checks the vector square root against
// float32(math.Sqrt(float64(x))), bit for bit, including negatives (NaN),
// -0, infinities and subnormals, at every length around the block, and in
// place.
func TestSqrt32MatchesScalar(t *testing.T) {
	for n := 0; n <= 70; n++ {
		src := operands32(n, 3)
		got, want := make([]float32, n), make([]float32, n)
		Sqrt32(got, src)
		sqrt32Go(want, src)
		for i := range got {
			if !sameBits32(got[i], want[i]) {
				t.Fatalf("n=%d [%d]: sqrt(%v) = %v, scalar %v", n, i, src[i], got[i], want[i])
			}
		}
		Sqrt32(src, src)
		for i := range src {
			if !sameBits32(src[i], want[i]) {
				t.Fatalf("n=%d [%d]: in place %v, scalar %v", n, i, src[i], want[i])
			}
		}
	}
}

func TestSqrt32RefusesAShortSource(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a short src was accepted")
		}
	}()
	Sqrt32(make([]float32, 4), make([]float32, 3))
}

func TestBin32RefusesBadInput(t *testing.T) {
	for _, f := range []func(){
		func() { Bin32(Add32, make([]float32, 4), make([]float32, 3), make([]float32, 4)) },
		func() { Bin32(Div32+1, make([]float32, 4), make([]float32, 4), make([]float32, 4)) },
		func() { bin32Go(-1, make([]float32, 1), make([]float32, 1), make([]float32, 1)) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("bad input was accepted")
				}
			}()
			f()
		}()
	}
}

// TestShortOperandWithSpareCapacityIsRefused is #130: a reslice x[:n] is
// checked against cap(x), not len(x), so an operand of length 3 and capacity 8
// passed with a dst of length 4 used to be read one element past its length.
// Every entry point must refuse it by length. The hidden element is a NaN so
// an over-read that slipped through would also show in the result.
func TestShortOperandWithSpareCapacityIsRefused(t *testing.T) {
	short := func() []float32 {
		s := make([]float32, 8)
		for i := range s {
			s[i] = float32(i + 1)
		}
		s[3] = float32(math.NaN())
		return s[:3] // len 3, cap 8
	}
	full := func() []float32 { return []float32{1, 2, 3, 4} }
	cases := map[string]func(){}
	for op := Add32; op <= Div32; op++ {
		op := op
		cases[fmt.Sprintf("Bin32 op %d short a", op)] = func() { Bin32(op, make([]float32, 4), short(), full()) }
		cases[fmt.Sprintf("Bin32 op %d short b", op)] = func() { Bin32(op, make([]float32, 4), full(), short()) }
		cases[fmt.Sprintf("bin32Go op %d short a", op)] = func() { bin32Go(op, make([]float32, 4), short(), full()) }
		cases[fmt.Sprintf("bin32Go op %d short b", op)] = func() { bin32Go(op, make([]float32, 4), full(), short()) }
	}
	cases["Sqrt32 short src"] = func() { Sqrt32(make([]float32, 4), short()) }
	cases["sqrt32Go short src"] = func() { sqrt32Go(make([]float32, 4), short()) }
	for name, f := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: a short operand with spare capacity was accepted", name)
				}
			}()
			f()
		}()
	}
}

// TestBin32OperandsLongerThanDst pins the contract the floats kernels need: they
// process len(a) elements and panic on a shorter operand, so Bin32 and Sqrt32
// must hand them operands of exactly len(dst) -- longer inputs are cut, shorter
// ones are refused by the wrapper (TestBin32RefusesBadInput), never by floats.
func TestBin32OperandsLongerThanDst(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7}
	b := []float32{7, 6, 5, 4, 3, 2, 1}
	for op := Add32; op <= Div32; op++ {
		got, want := make([]float32, 4), make([]float32, 4)
		Bin32(op, got, a, b)
		bin32Go(op, want, a, b)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("op %d lane %d: %v != %v", op, i, got[i], want[i])
			}
		}
	}
	got, want := make([]float32, 4), make([]float32, 4)
	Sqrt32(got, a)
	sqrt32Go(want, a)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("sqrt lane %d: %v != %v", i, got[i], want[i])
		}
	}
}
