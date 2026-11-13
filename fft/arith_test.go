package fft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// TestArithmeticOptions: Exact and FMA reach go-fft. Exact gives today's
// default bits (go-fft's default is its reference arithmetic); FMA stays
// within the transform's error bound of the default; asking for both is
// refused; and a repeated axis (the sequential path) carries them too.
func TestArithmeticOptions(t *testing.T) {
	// Today Exact and FMA rarely change bits that a test can see (Exact is
	// the default, and FMA acts only on amd64 with FMA3), so the hand-over to
	// go-fft is asserted on the Options it builds.
	for _, o := range []Options{{Exact: true}, {FMA: true}} {
		if g := o.engine([]int{0}, Ortho); g.Exact != o.Exact || g.FMA != o.FMA || g.Norm != Ortho || len(g.Axes) != 1 {
			t.Fatalf("engine(%+v) = %+v", o, g)
		}
	}
	x, err := ndarray.Arange(0, 48, 1)
	if err != nil {
		t.Fatal(err)
	}
	x, _ = x.Sin().Reshape(6, 8)
	for _, c := range []struct {
		name string
		f    func(*ndarray.Array, Options) (*ndarray.Array, error)
		axes []int
	}{
		{"FFTN", FFTNWith, nil},
		{"RFFTN", RFFTNWith, nil},
		{"FFTN repeated axis", FFTNWith, []int{1, 1}},
	} {
		def, err := c.f(x, Options{Axes: c.axes})
		if err != nil {
			t.Fatal(err)
		}
		exact, err := c.f(x, Options{Axes: c.axes, Exact: true})
		if err != nil {
			t.Fatal(err)
		}
		fma, err := c.f(x, Options{Axes: c.axes, FMA: true})
		if err != nil {
			t.Fatal(err)
		}
		d, _ := ndarray.Data[complex128](def)
		e, _ := ndarray.Data[complex128](exact)
		f, _ := ndarray.Data[complex128](fma)
		peak := 0.0
		for _, v := range d {
			peak = math.Max(peak, cmplx.Abs(v))
		}
		for i := range d {
			if e[i] != d[i] {
				t.Fatalf("%s: Exact element %d = %v, default %v", c.name, i, e[i], d[i])
			}
			if cmplx.Abs(f[i]-d[i]) > 64*0x1p-52*peak {
				t.Fatalf("%s: FMA element %d = %v, default %v", c.name, i, f[i], d[i])
			}
		}
		if _, err := c.f(x, Options{Axes: c.axes, Exact: true, FMA: true}); !errors.Is(err, ErrArgs) {
			t.Fatalf("%s: Exact and FMA together gave %v, want ErrArgs", c.name, err)
		}
	}
}
