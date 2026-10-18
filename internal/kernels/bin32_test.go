package kernels

import (
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
