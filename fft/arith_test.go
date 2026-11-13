package fft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	gofft "github.com/go-fft/fft"
	"github.com/go-ndarray/ndarray"
)

// TestArithmeticReachesGoFFT: Exact and FMA arrive in the go-fft Options of
// every kind of transform (c2c, r2c, c2r, herm, iherm) and of the repeated-axis
// path, read from what is actually handed to go-fft. Today Exact equals the
// default and FMA acts only on amd64 with FMA3, so the results alone could not
// show a dropped field.
func TestArithmeticReachesGoFFT(t *testing.T) {
	var got []gofft.Options
	defer func(h func(gofft.Options)) { handed = h }(handed)
	handed = func(o gofft.Options) { got = append(got, o) }
	x, err := ndarray.Arange(0, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	m, _ := x.Reshape(4, 4)
	for _, o := range []Options{{Exact: true}, {FMA: true}} {
		for _, c := range []struct {
			name string
			f    func(*ndarray.Array, Options) (*ndarray.Array, error)
			a    *ndarray.Array
			axes []int
		}{
			{"FFT", FFTWith, x, nil}, {"RFFT", RFFTWith, x, nil}, {"IRFFT", IRFFTWith, x, nil},
			{"HFFT", HFFTWith, x, nil}, {"IHFFT", IHFFTWith, x, nil},
			{"FFTN repeated axis", FFTNWith, m, []int{1, 1}},
		} {
			got = got[:0]
			o.Axes = c.axes
			if _, err := c.f(c.a, o); err != nil {
				t.Fatal(err)
			}
			if len(got) == 0 {
				t.Fatalf("%s %+v: nothing reached go-fft", c.name, o)
			}
			for _, g := range got {
				if g.Exact != o.Exact || g.FMA != o.FMA {
					t.Fatalf("%s %+v: go-fft got Exact=%v FMA=%v", c.name, o, g.Exact, g.FMA)
				}
			}
		}
	}
}

// TestArithmeticOptions: Exact and FMA reach go-fft. Exact gives today's
// default bits (go-fft's default is its reference arithmetic); FMA stays
// within the transform's error bound of the default; asking for both is
// refused; and a repeated axis (the sequential path) carries them too.
func TestArithmeticOptions(t *testing.T) {
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
