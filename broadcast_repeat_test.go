package ndarray

import (
	"testing"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// TestBroadcastRepeat checks the streamed broadcast (a full operand and a
// repeated one, see repeatOperands) against an element-by-element oracle, for
// the non-commutative ops in both operand orders, several suffix shapes, a
// repeated operand too short to stream, and with the block loop forced onto
// several cores.
func TestBroadcastRepeat(t *testing.T) {
	grid := func(seed int, shape ...int) *Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		d := make([]float64, n)
		for i := range d {
			d[i] = float64((i*13+seed)%29) - 14.5
		}
		return mustArr(t, ok(FromData(d, shape...)))
	}
	cases := []struct {
		name       string
		full, rep  *Array
		streamable bool
	}{
		{"matrix, row", grid(1, 9, 70), grid(2, 1, 70), true},
		{"matrix, 1-D", grid(3, 5, 80), grid(4, 80), true},
		{"3-D, 2-D", grid(5, 4, 3, 30), grid(6, 3, 30), true},
		{"3-D, row of 1-D", grid(7, 2, 3, 100), grid(8, 1, 1, 100), true},
		{"short row", grid(9, 6, 10), grid(10, 1, 10), false},
	}
	ops := []struct {
		name string
		f    func(x, y *Array) (*Array, error)
		g    func(x, y float64) float64
	}{
		{"sub", (*Array).Sub, func(x, y float64) float64 { return x - y }},
		{"div", (*Array).Div, func(x, y float64) float64 { return x / y }},
	}
	run := func() {
		for _, c := range cases {
			if _, _, _, got := repeatOperands(c.full, c.rep, c.full.shape); got != c.streamable {
				t.Errorf("%s: streamable = %v, want %v", c.name, got, c.streamable)
			}
			for _, op := range ops {
				for _, swap := range []bool{false, true} {
					x, y := c.full, c.rep
					if swap {
						x, y = y, x
					}
					got, err := op.f(x, y)
					if err != nil {
						t.Fatal(err)
					}
					if !sameShape(got.shape, c.full.shape) {
						t.Fatalf("%s %s: shape %v", c.name, op.name, got.shape)
					}
					m := c.rep.Size()
					for i, v := range got.data {
						fv, rv := c.full.data[i], c.rep.data[i%m]
						want := op.g(fv, rv)
						if swap {
							want = op.g(rv, fv)
						}
						if v != want {
							t.Fatalf("%s %s swap=%v [%d]: %v, want %v", c.name, op.name, swap, i, v, want)
						}
					}
				}
			}
		}
	}
	run()
	saved := kernels.ParThreshold
	kernels.ParThreshold = 64
	defer func() { kernels.ParThreshold = saved }()
	run()
}
