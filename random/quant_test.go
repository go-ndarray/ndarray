package random

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// TestQuantifyLibm measures, on 10^6 draws per sampler, how many values
// differ from NumPy's and by how much. It needs NumPy's output, written by
// testdata/quant.py (not embedded: 180 MB), and runs only when
// RANDOM_QUANT_DIR names its directory; docs/random.md quotes the result.
func TestQuantifyLibm(t *testing.T) {
	dir := os.Getenv("RANDOM_QUANT_DIR")
	if dir == "" {
		t.Skip("RANDOM_QUANT_DIR not set")
	}
	const n = 1_000_000
	g := func() *Generator { return DefaultRNG(0) }
	l := func() *RandomState { return NewRandomState(0) }
	asF := func(a *ndarray.Array, err error) []float64 {
		if err != nil {
			t.Fatal(err)
		}
		d, _ := ndarray.Data[float64](a.AsType(ndarray.Float64))
		return d
	}
	jobs := []struct {
		name string
		f    func() []float64
	}{
		{"legacy_randn", func() []float64 { return asF(l().Randn(n)) }},
		{"legacy_normal", func() []float64 { return asF(l().Normal(1, 3, n)) }},
		{"legacy_gamma3", func() []float64 { return asF(l().StandardGamma(3, n)) }},
		{"legacy_exp", func() []float64 { return asF(l().StandardExponential(n)) }},
		{"gamma05", func() []float64 { return asF(g().StandardGamma(0.5, n)) }},
		{"gamma3", func() []float64 { return asF(g().StandardGamma(3, n)) }},
		{"beta23", func() []float64 { return asF(g().Beta(2, 3, n)) }},
		{"beta05", func() []float64 { return asF(g().Beta(0.5, 0.5, n)) }},
		{"lognormal", func() []float64 { return asF(g().Lognormal(0.5, 0.25, n)) }},
		{"laplace", func() []float64 { return asF(g().Laplace(0, 1, n)) }},
		{"logistic", func() []float64 { return asF(g().Logistic(0, 1, n)) }},
		{"gumbel", func() []float64 { return asF(g().Gumbel(0, 1, n)) }},
		{"exp_inv", func() []float64 { return asF(g().StandardExponentialOf(ndarray.Float64, "inv", n)) }},
		{"pareto", func() []float64 { return asF(g().Pareto(3, n)) }},
		{"weibull", func() []float64 { return asF(g().Weibull(2, n)) }},
		{"vonmises", func() []float64 { return asF(g().VonMises(0.5, 2, n)) }},
		{"chisquare", func() []float64 { return asF(g().ChiSquare(3, n)) }},
		{"standard_t", func() []float64 { return asF(g().StandardT(4, n)) }},
		{"normal", func() []float64 { return asF(g().StandardNormal(n)) }},
		{"binomial", func() []float64 { return asF(g().Binomial(1000, 0.4, n)) }},
		{"poisson", func() []float64 { return asF(g().Poisson(50, n)) }},
		{"poisson3", func() []float64 { return asF(g().Poisson(3, n)) }},
		{"geometric", func() []float64 { return asF(g().Geometric(0.05, n)) }},
	}
	for _, j := range jobs {
		raw, err := os.ReadFile(filepath.Join(dir, j.name+".bin"))
		if err != nil {
			t.Fatal(err)
		}
		got := j.f()
		diff, maxU := 0, int64(0)
		for i := range got {
			w := math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
			if got[i] != w {
				diff++
				maxU = max(maxU, ulps(got[i], w, false))
			}
		}
		t.Logf("%-14s %7d of %d differ, max %d ULP", j.name, diff, len(got), maxU)
	}
}
