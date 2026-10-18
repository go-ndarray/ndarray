package kernels

import (
	"math"
	"testing"
)

// TestAddBlocks8F32MatchesScalar: the kernel must give the scalar loop's
// sums bit for bit (one float32 addition per lane per block, in order), for
// every number of blocks up to 40 and values chosen to round, to overflow,
// to underflow and to carry NaN, the infinities and -0.
func TestAddBlocks8F32MatchesScalar(t *testing.T) {
	special := []float32{
		0, float32(math.Copysign(0, -1)), 1, -1, 0.1, -0.3, 3.4e38, -3.4e38,
		1e-45, -1e-45, 1.17549435e-38, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), 16777216, 1, 0.5,
	}
	x := uint32(7)
	next := func() float32 {
		x = x*1664525 + 1013904223
		if x%5 == 0 {
			return special[int(x>>8)%len(special)]
		}
		return float32(int32(x)) / float32(1<<20)
	}
	for blocks := 1; blocks <= 40; blocks++ {
		a := make([]float32, 8*blocks)
		for i := range a {
			a[i] = next()
		}
		var r0 [8]float32
		for j := range r0 {
			r0[j] = next()
		}
		got, want := r0, r0
		AddBlocks8F32(&got, a)
		addBlocks8F32Go(&want, a)
		for j := range got {
			g, w := got[j], want[j]
			if g != g && w != w {
				continue // both NaN: the payload is not what is tested
			}
			if math.Float32bits(g) != math.Float32bits(w) {
				t.Fatalf("blocks=%d lane %d: kernel %v (%#08x), scalar %v (%#08x)",
					blocks, j, g, math.Float32bits(g), w, math.Float32bits(w))
			}
		}
	}
}

func TestAddBlocks8F32Refusals(t *testing.T) {
	var r [8]float32
	AddBlocks8F32(&r, nil) // no blocks: nothing to add
	if r != [8]float32{} {
		t.Errorf("an empty a changed r: %v", r)
	}
	defer func() {
		if recover() == nil {
			t.Error("a length that is not a multiple of 8 did not panic")
		}
	}()
	AddBlocks8F32(&r, make([]float32, 9))
}
