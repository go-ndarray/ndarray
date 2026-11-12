package random

import (
	"testing"

	"github.com/go-ndarray/ndarray"
)

// TestChoiceAllocsConstant: the gather of Choice allocates nothing per drawn
// element (#103: it was two per element through reflect.Value.Slice), so 100
// and 10 000 draws cost the same number of allocations, along axis 0 of a 1-D
// array and along an inner axis of a 2-D one.
func TestChoiceAllocsConstant(t *testing.T) {
	a, err := ndarray.Arange(0, 1000, 1)
	if err != nil {
		t.Fatal(err)
	}
	m, err := a.Reshape(10, 100)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		a    *ndarray.Array
		axis int
	}{{"1-D", a, 0}, {"2-D axis 1", m, 1}} {
		allocs := func(size int) float64 {
			g := DefaultRNG(1)
			return testing.AllocsPerRun(20, func() {
				if _, err := g.Choice(c.a, ChoiceOptions{Axis: c.axis}, size); err != nil {
					t.Fatal(err)
				}
			})
		}
		if small, large := allocs(100), allocs(10000); large != small {
			t.Errorf("%s: %v allocations for 100 draws, %v for 10000: the gather allocates per element", c.name, small, large)
		}
	}
}
