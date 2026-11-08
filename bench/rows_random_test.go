package bench

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// Random rows. Each draws 100 000 values (20 000 vectors for Dirichlet and
// Multinomial) from a generator built once outside the timed loop, on both
// sides: numpy.random.default_rng(1) / random.DefaultRNG(1) for Generator, and
// np.random.RandomState(1) / random.NewRandomState(1) for the legacy
// RandomState, np.random.seed(1) for the package-level functions. The streams
// are numpy's own, bit for bit (random/numpy_test.go); only speed is compared.

const (
	rn = 100_000
	rv = 20_000
)

var dChoice = dvec(0, 1, 1000, 0)

// gen is a Generator row: py is the draw as a method call on g.
func gen(name, py string, body func(g *random.Generator) *ndarray.Array) row {
	return row{name: "random.Generator." + name, fn: "random.Generator." + name, ref: "numpy Generator." + py,
		call: "(lambda g: lambda: g." + py + ")(np.random.default_rng(1))",
		body: func([]*ndarray.Array) func() {
			g := random.DefaultRNG(1)
			return func() { sinkA = body(g) }
		}}
}

func gv(a *ndarray.Array, err error) *ndarray.Array { return must(a, err) }

func genRows() []row {
	rs := []row{
		gen("Beta", "beta(2.0, 3.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Beta(2, 3, rn)) }),
		gen("Binomial", "binomial(100, 0.3, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Binomial(100, 0.3, rn)) }),
		gen("ChiSquare", "chisquare(3.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.ChiSquare(3, rn)) }),
		gen("ChoiceN", "choice(1000, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.ChoiceN(1000, random.ChoiceOptions{}, rn)) }),
		gen("Dirichlet", "dirichlet([1.0, 2.0, 3.0], 20000)", func(g *random.Generator) *ndarray.Array { return gv(g.Dirichlet([]float64{1, 2, 3}, rv)) }),
		gen("Exponential", "exponential(1.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Exponential(1.5, rn)) }),
		gen("F", "f(5.0, 10.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.F(5, 10, rn)) }),
		gen("Gamma", "gamma(2.0, 1.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Gamma(2, 1.5, rn)) }),
		gen("Geometric", "geometric(0.2, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Geometric(0.2, rn)) }),
		gen("Gumbel", "gumbel(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Gumbel(0, 1, rn)) }),
		gen("Integers", "integers(0, 1000, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Integers(0, 1000, rn)) }),
		gen("IntegersOf", "integers(0, 1000, 100000, dtype=np.int32)", func(g *random.Generator) *ndarray.Array {
			return gv(g.IntegersOf(ndarray.Int32, 0, 1000, false, rn))
		}),
		gen("IntegersUint64", "integers(0, 1000, 100000, dtype=np.uint64)", func(g *random.Generator) *ndarray.Array {
			return gv(g.IntegersUint64(0, 1000, false, rn))
		}),
		gen("Laplace", "laplace(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Laplace(0, 1, rn)) }),
		gen("Logistic", "logistic(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Logistic(0, 1, rn)) }),
		gen("Lognormal", "lognormal(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Lognormal(0, 1, rn)) }),
		gen("Logseries", "logseries(0.6, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Logseries(0.6, rn)) }),
		gen("Multinomial", "multinomial(100, [0.2, 0.3, 0.5], 20000)", func(g *random.Generator) *ndarray.Array {
			return gv(g.Multinomial(100, []float64{0.2, 0.3, 0.5}, rv))
		}),
		gen("NegativeBinomial", "negative_binomial(5.0, 0.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.NegativeBinomial(5, 0.5, rn)) }),
		gen("NoncentralChiSquare", "noncentral_chisquare(3.0, 2.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.NoncentralChiSquare(3, 2, rn)) }),
		gen("NoncentralF", "noncentral_f(5.0, 10.0, 2.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.NoncentralF(5, 10, 2, rn)) }),
		gen("Normal", "normal(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Normal(0, 1, rn)) }),
		gen("Pareto", "pareto(3.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Pareto(3, rn)) }),
		gen("Permutation", "permutation(100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Permutation(rn)) }),
		gen("Poisson", "poisson(5.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Poisson(5, rn)) }),
		gen("Power", "power(3.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Power(3, rn)) }),
		gen("Random", "random(100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Random(rn)) }),
		gen("RandomOf", "random(100000, dtype=np.float32)", func(g *random.Generator) *ndarray.Array { return gv(g.RandomOf(ndarray.Float32, rn)) }),
		gen("Rayleigh", "rayleigh(1.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Rayleigh(1.5, rn)) }),
		gen("StandardCauchy", "standard_cauchy(100000)", func(g *random.Generator) *ndarray.Array { return gv(g.StandardCauchy(rn)) }),
		gen("StandardExponential", "standard_exponential(100000)", func(g *random.Generator) *ndarray.Array { return gv(g.StandardExponential(rn)) }),
		gen("StandardExponentialOf", "standard_exponential(100000, dtype=np.float32, method='zig')", func(g *random.Generator) *ndarray.Array {
			return gv(g.StandardExponentialOf(ndarray.Float32, "zig", rn))
		}),
		gen("StandardGamma", "standard_gamma(2.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.StandardGamma(2, rn)) }),
		gen("StandardGammaOf", "standard_gamma(2.0, 100000, dtype=np.float32)", func(g *random.Generator) *ndarray.Array {
			return gv(g.StandardGammaOf(ndarray.Float32, 2, rn))
		}),
		gen("StandardNormal", "standard_normal(100000)", func(g *random.Generator) *ndarray.Array { return gv(g.StandardNormal(rn)) }),
		gen("StandardNormalOf", "standard_normal(100000, dtype=np.float32)", func(g *random.Generator) *ndarray.Array {
			return gv(g.StandardNormalOf(ndarray.Float32, rn))
		}),
		gen("StandardT", "standard_t(5.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.StandardT(5, rn)) }),
		gen("Triangular", "triangular(0.0, 1.0, 3.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Triangular(0, 1, 3, rn)) }),
		gen("Uniform", "uniform(0.0, 1.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Uniform(0, 1, rn)) }),
		gen("VonMises", "vonmises(0.0, 2.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.VonMises(0, 2, rn)) }),
		gen("Wald", "wald(1.0, 2.0, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Wald(1, 2, rn)) }),
		gen("Weibull", "weibull(1.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Weibull(1.5, rn)) }),
		gen("Zipf", "zipf(2.5, 100000)", func(g *random.Generator) *ndarray.Array { return gv(g.Zipf(2.5, rn)) }),
	}
	// Rows that take an array.
	withArr := func(name, py string, body func(g *random.Generator, a *ndarray.Array) *ndarray.Array) row {
		return row{name: "random.Generator." + name, fn: "random.Generator." + name, ref: "numpy Generator." + py, args: []ds{dChoice},
			call: "(lambda g: lambda x: g." + py + ")(np.random.default_rng(1))",
			body: func(a []*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				return func() { sinkA = body(g, a[0]) }
			}}
	}
	rs = append(rs,
		withArr("Choice", "choice(x, 100000)", func(g *random.Generator, a *ndarray.Array) *ndarray.Array {
			return gv(g.Choice(a, random.ChoiceOptions{}, rn))
		}),
		withArr("PermutationOf", "permutation(x, axis=0)", func(g *random.Generator, a *ndarray.Array) *ndarray.Array { return gv(g.PermutationOf(a, 0)) }),
		withArr("Permuted", "permuted(x)", func(g *random.Generator, a *ndarray.Array) *ndarray.Array { return gv(g.Permuted(a)) }),
		withArr("PermutedAxis", "permuted(x, axis=0)", func(g *random.Generator, a *ndarray.Array) *ndarray.Array { return gv(g.PermutedAxis(a, 0)) }),
		// Shuffle is in place, on a copy private to each side.
		row{name: "random.Generator.Shuffle", fn: "random.Generator.Shuffle", ref: "numpy Generator.shuffle (in place)", args: []ds{dChoice},
			call: "(lambda g: lambda x: g.shuffle(x.copy()))(np.random.default_rng(1))",
			body: func(a []*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				return func() { chk(g.Shuffle(a[0].Copy(), 0)) }
			}},
		row{name: "random.Generator.ShuffleInt64", fn: "random.Generator.ShuffleInt64", ref: "numpy Generator.shuffle of 100000 int64 (in place)",
			call: "(lambda g, x: lambda: g.shuffle(x))(np.random.default_rng(1), np.arange(100000, dtype=np.int64))",
			body: func([]*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				d := make([]int64, rn)
				for i := range d {
					d[i] = int64(i)
				}
				return func() { g.ShuffleInt64(d) }
			}},
		row{name: "random.Generator.FillIntN", fn: "random.Generator.FillIntN", ref: "numpy Generator.integers(0, 1000, 100000) (allocates; Go fills dst)",
			call: "(lambda g: lambda: g.integers(0, 1000, 100000))(np.random.default_rng(1))",
			body: func([]*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				d := make([]int64, rn)
				return func() { chk(g.FillIntN(d, 1000)) }
			}},
		row{name: "random.Generator.FillUniform", fn: "random.Generator.FillUniform", ref: "numpy Generator.uniform(0, 1, 100000) (allocates; Go fills dst)",
			call: "(lambda g: lambda: g.uniform(0.0, 1.0, 100000))(np.random.default_rng(1))",
			body: func([]*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				d := make([]float64, rn)
				return func() { chk(g.FillUniform(d, 0, 1)) }
			}},
		row{name: "random.Generator.Bytes", fn: "random.Generator.Bytes", ref: "numpy Generator.bytes(1000000)",
			call: "(lambda g: lambda: g.bytes(1000000))(np.random.default_rng(1))",
			body: func([]*ndarray.Array) func() {
				g := random.DefaultRNG(1)
				return func() { sinkI = len(g.Bytes(1_000_000)) }
			}},
	)
	return rs
}

// rs is a RandomState row: np.random.RandomState(1).<py>.
func rst(name, py string, body func(r *random.RandomState) *ndarray.Array) row {
	return row{name: "random.RandomState." + name, fn: "random.RandomState." + name, ref: "numpy RandomState." + py,
		call: "(lambda r: lambda: r." + py + ")(np.random.RandomState(1))",
		body: func([]*ndarray.Array) func() {
			r := random.NewRandomState(1)
			return func() { sinkA = body(r) }
		}}
}

func legacyRows() []row {
	rs := []row{
		rst("Binomial", "binomial(100, 0.3, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Binomial(100, 0.3, rn)) }),
		rst("ChiSquare", "chisquare(3.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.ChiSquare(3, rn)) }),
		rst("ChoiceN", "choice(1000, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.ChoiceN(1000, random.ChoiceOptions{}, rn)) }),
		rst("Exponential", "exponential(1.5, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Exponential(1.5, rn)) }),
		rst("Gamma", "gamma(2.0, 1.5, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Gamma(2, 1.5, rn)) }),
		rst("Lognormal", "lognormal(0.0, 1.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Lognormal(0, 1, rn)) }),
		rst("Multinomial", "multinomial(100, [0.2, 0.3, 0.5], 20000)", func(r *random.RandomState) *ndarray.Array {
			return gv(r.Multinomial(100, []float64{0.2, 0.3, 0.5}, rv))
		}),
		rst("Normal", "normal(0.0, 1.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Normal(0, 1, rn)) }),
		rst("Permutation", "permutation(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Permutation(rn)) }),
		rst("Poisson", "poisson(5.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Poisson(5, rn)) }),
		rst("Rand", "rand(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Rand(rn)) }),
		rst("Randint", "randint(0, 1000, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Randint(0, 1000, rn)) }),
		rst("RandintOf", "randint(0, 1000, 100000, dtype=np.int32)", func(r *random.RandomState) *ndarray.Array {
			return gv(r.RandintOf(ndarray.Int32, 0, 1000, rn))
		}),
		rst("Randn", "randn(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Randn(rn)) }),
		rst("RandomSample", "random_sample(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.RandomSample(rn)) }),
		rst("StandardExponential", "standard_exponential(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.StandardExponential(rn)) }),
		rst("StandardGamma", "standard_gamma(2.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.StandardGamma(2, rn)) }),
		rst("StandardNormal", "standard_normal(100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.StandardNormal(rn)) }),
		rst("Uniform", "uniform(0.0, 1.0, 100000)", func(r *random.RandomState) *ndarray.Array { return gv(r.Uniform(0, 1, rn)) }),
	}
	rs = append(rs,
		row{name: "random.RandomState.Choice", fn: "random.RandomState.Choice", ref: "numpy RandomState.choice(x, 100000)", args: []ds{dChoice},
			call: "(lambda r: lambda x: r.choice(x, 100000))(np.random.RandomState(1))",
			body: func(a []*ndarray.Array) func() {
				r := random.NewRandomState(1)
				return func() { sinkA = gv(r.Choice(a[0], random.ChoiceOptions{}, rn)) }
			}},
		row{name: "random.RandomState.PermutationOf", fn: "random.RandomState.PermutationOf", ref: "numpy RandomState.permutation(x)", args: []ds{dChoice},
			call: "(lambda r: lambda x: r.permutation(x))(np.random.RandomState(1))",
			body: func(a []*ndarray.Array) func() {
				r := random.NewRandomState(1)
				return func() { sinkA = gv(r.PermutationOf(a[0])) }
			}},
		row{name: "random.RandomState.Shuffle", fn: "random.RandomState.Shuffle", ref: "numpy RandomState.shuffle (in place, on a copy)", args: []ds{dChoice},
			call: "(lambda r: lambda x: r.shuffle(x.copy()))(np.random.RandomState(1))",
			body: func(a []*ndarray.Array) func() {
				r := random.NewRandomState(1)
				return func() { chk(r.Shuffle(a[0].Copy())) }
			}},
		row{name: "random.RandomState.ShuffleInt64", fn: "random.RandomState.ShuffleInt64", ref: "numpy RandomState.shuffle of 100000 int64 (in place)",
			call: "(lambda r, x: lambda: r.shuffle(x))(np.random.RandomState(1), np.arange(100000, dtype=np.int64))",
			body: func([]*ndarray.Array) func() {
				r := random.NewRandomState(1)
				d := make([]int64, rn)
				return func() { r.ShuffleInt64(d) }
			}},
		row{name: "random.RandomState.FillIntN", fn: "random.RandomState.FillIntN", ref: "numpy RandomState.randint(0, 1000, 100000) (allocates; Go fills dst)",
			call: "(lambda r: lambda: r.randint(0, 1000, 100000))(np.random.RandomState(1))",
			body: func([]*ndarray.Array) func() {
				r := random.NewRandomState(1)
				d := make([]int64, rn)
				return func() { chk(r.FillIntN(d, 1000)) }
			}},
		row{name: "random.RandomState.FillUniform", fn: "random.RandomState.FillUniform", ref: "numpy RandomState.uniform(0, 1, 100000) (allocates; Go fills dst)",
			call: "(lambda r: lambda: r.uniform(0.0, 1.0, 100000))(np.random.RandomState(1))",
			body: func([]*ndarray.Array) func() {
				r := random.NewRandomState(1)
				d := make([]float64, rn)
				return func() { chk(r.FillUniform(d, 0, 1)) }
			}},
		row{name: "random.RandomState.Bytes", fn: "random.RandomState.Bytes", ref: "numpy RandomState.bytes(1000000)",
			call: "(lambda r: lambda: r.bytes(1000000))(np.random.RandomState(1))",
			body: func([]*ndarray.Array) func() {
				r := random.NewRandomState(1)
				return func() { sinkI = len(r.Bytes(1_000_000)) }
			}},
	)
	return rs
}

// pkg is a package-level function: np.random.<py> on the shared legacy state.
func pkg(name, py string, body func() *ndarray.Array) row {
	return row{name: "random." + name, fn: "random." + name, ref: "numpy np.random." + py,
		call: "(lambda _: lambda: np.random." + py + ")(np.random.seed(1))",
		body: func([]*ndarray.Array) func() {
			random.Seed(1)
			return func() { sinkA = body() }
		}}
}

func packageRows() []row {
	return []row{
		pkg("Binomial", "binomial(100, 0.3, 100000)", func() *ndarray.Array { return gv(random.Binomial(100, 0.3, rn)) }),
		pkg("Normal", "normal(0.0, 1.0, 100000)", func() *ndarray.Array { return gv(random.Normal(0, 1, rn)) }),
		pkg("Permutation", "permutation(100000)", func() *ndarray.Array { return gv(random.Permutation(rn)) }),
		pkg("Poisson", "poisson(5.0, 100000)", func() *ndarray.Array { return gv(random.Poisson(5, rn)) }),
		pkg("Rand", "rand(100000)", func() *ndarray.Array { return gv(random.Rand(rn)) }),
		pkg("Randint", "randint(0, 1000, 100000)", func() *ndarray.Array { return gv(random.Randint(0, 1000, rn)) }),
		pkg("Randn", "randn(100000)", func() *ndarray.Array { return gv(random.Randn(rn)) }),
		pkg("RandomSample", "random_sample(100000)", func() *ndarray.Array { return gv(random.RandomSample(rn)) }),
		pkg("Uniform", "uniform(0.0, 1.0, 100000)", func() *ndarray.Array { return gv(random.Uniform(0, 1, rn)) }),
		{name: "random.Choice", fn: "random.Choice", ref: "numpy np.random.choice(x, 100000)", args: []ds{dChoice},
			call: "(lambda _: lambda x: np.random.choice(x, 100000))(np.random.seed(1))",
			body: func(a []*ndarray.Array) func() {
				random.Seed(1)
				return func() { sinkA = gv(random.Choice(a[0], random.ChoiceOptions{}, rn)) }
			}},
		{name: "random.Shuffle", fn: "random.Shuffle", ref: "numpy np.random.shuffle (in place, on a copy)", args: []ds{dChoice},
			call: "(lambda _: lambda x: np.random.shuffle(x.copy()))(np.random.seed(1))",
			body: func(a []*ndarray.Array) func() {
				random.Seed(1)
				return func() { chk(random.Shuffle(a[0].Copy())) }
			}},
		{name: "random.RandomRaw", fn: "random.RandomRaw", ref: "numpy PCG64.random_raw(100000)",
			call: "(lambda b: lambda: b.random_raw(100000))(np.random.PCG64(1))",
			body: func([]*ndarray.Array) func() {
				bg := random.NewPCG64(1)
				return func() { sinkI = len(random.RandomRaw(bg, rn)) }
			}},
	}
}

var _ = fmt.Sprint
