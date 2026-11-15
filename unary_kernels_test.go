package ndarray

import (
	"math"
	"math/rand"
	"testing"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// unaryEdge is the table every float64 unary kernel is checked on.
func unaryEdge() []float64 {
	nan := math.NaN()
	snan := math.Float64frombits(0x7FF0000000000001) // signalling NaN
	two52, two63 := math.Ldexp(1, 52), math.Ldexp(1, 63)
	v := []float64{
		0, math.Copysign(0, -1), math.Inf(1), math.Inf(-1), nan, -nan, snan,
		math.Float64frombits(0xFFF8000000000123),
		5e-324, -5e-324, 2.2250738585072009e-308, -2.2250738585072014e-308,
		math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64,
		0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994,
		1, -1, 0.1, -0.1, 3.7, -3.7, 1e-10, -1e-10, 1e10, -1e10,
		1e154, -1e154, 1e155, 1e-155,
	}
	for _, b := range []float64{two52, two63, math.Ldexp(1, 53), math.Ldexp(1, 62), math.Ldexp(1, 64)} {
		for _, s := range []float64{1, -1} {
			v = append(v, s*b, s*math.Nextafter(b, 0), s*math.Nextafter(b, math.Inf(1)),
				s*(b+0.5), s*(b-0.5), s*(b/2+0.5))
		}
	}
	return v
}

func sameBits(got, want float64) bool {
	if math.IsNaN(want) {
		return math.IsNaN(got) // a signalling NaN may come back quieted
	}
	return math.Float64bits(got) == math.Float64bits(want)
}

// TestUnaryKernelsBitIdentical checks Abs, Neg, Square, Floor and Ceil against
// the math expression they replaced, on lengths 0..33 and one above the
// parallel threshold, cut from the edge table and random values.
func TestUnaryKernelsBitIdentical(t *testing.T) {
	ops := []struct {
		name string
		do   func(*Array) *Array
		ref  func(float64) float64
	}{
		{"Abs", (*Array).Abs, math.Abs},
		{"Neg", (*Array).Neg, func(x float64) float64 { return -x }},
		{"Square", (*Array).Square, func(x float64) float64 { return x * x }},
		{"Floor", (*Array).Floor, math.Floor},
		{"Ceil", (*Array).Ceil, math.Ceil},
	}
	rng := rand.New(rand.NewSource(1))
	pool := unaryEdge()
	for i := 0; i < 4096; i++ {
		pool = append(pool, math.Float64frombits(rng.Uint64()), (rng.Float64()-0.5)*1e6,
			math.Round(rng.Float64()*2000-1000)/2)
	}
	lengths := make([]int, 0, 35)
	for n := 0; n <= 33; n++ {
		lengths = append(lengths, n)
	}
	lengths = append(lengths, kernels.ParThreshold+37)
	for _, op := range ops {
		for _, n := range lengths {
			for off := 0; off < 3; off++ { // different alignments of the table
				src := make([]float64, n)
				for i := range src {
					src[i] = pool[(off*7+i*(1+off))%len(pool)]
				}
				if n > 64 {
					copy(src, pool) // the whole edge table, in the parallel case
				}
				a, err := FromData(append([]float64(nil), src...), n)
				if err != nil {
					t.Fatal(err)
				}
				got := op.do(a).materialize()
				if len(got) != n {
					t.Fatalf("%s n=%d: len %d", op.name, n, len(got))
				}
				for i, x := range src {
					if w := op.ref(x); !sameBits(got[i], w) {
						t.Fatalf("%s n=%d [%d] %#x: got %#x want %#x", op.name, n, i,
							math.Float64bits(x), math.Float64bits(got[i]), math.Float64bits(w))
					}
				}
				in := a.materialize()
				for i := range src { // the input is not modified
					if !sameBits(in[i], src[i]) {
						t.Fatalf("%s n=%d: input modified at %d", op.name, n, i)
					}
				}
			}
		}
	}
}

// TestUnaryKernelsStrided checks the non-contiguous path (a transposed view).
func TestUnaryKernelsStrided(t *testing.T) {
	src := unaryEdge()
	n := len(src) / 2 * 2
	a, err := FromData(append([]float64(nil), src[:n]...), 2, n/2)
	if err != nil {
		t.Fatal(err)
	}
	tr := a.Transpose()
	view := tr.materialize()
	for name, c := range map[string]struct {
		got []float64
		ref func(float64) float64
	}{
		"Abs":    {tr.Abs().materialize(), math.Abs},
		"Neg":    {tr.Neg().materialize(), func(x float64) float64 { return -x }},
		"Square": {tr.Square().materialize(), func(x float64) float64 { return x * x }},
		"Floor":  {tr.Floor().materialize(), math.Floor},
		"Ceil":   {tr.Ceil().materialize(), math.Ceil},
	} {
		for i, x := range view {
			if !sameBits(c.got[i], c.ref(x)) {
				t.Fatalf("%s strided [%d]: got %#x want %#x", name, i,
					math.Float64bits(c.got[i]), math.Float64bits(c.ref(x)))
			}
		}
	}
}
