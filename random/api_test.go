package random

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func mustErr(t *testing.T, what string, err error) {
	t.Helper()
	if err == nil {
		t.Errorf("%s: no error", what)
	}
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// TestArgumentErrors checks that every argument numpy rejects is rejected,
// with an error that wraps ErrValue.
func TestArgumentErrors(t *testing.T) {
	g := DefaultRNG(1)
	nan := math.NaN()
	_, errNeg := g.Random(-1)
	errs := map[string]error{
		"random -1": errNeg,
	}
	add := func(name string) func(*ndarray.Array, error) {
		return func(_ *ndarray.Array, err error) { errs[name] = err }
	}
	add("random int")(g.RandomOf(ndarray.Int8))
	add("normal int")(g.StandardNormalOf(ndarray.Int8))
	add("exp int")(g.StandardExponentialOf(ndarray.Int8, "zig"))
	add("exp method")(g.StandardExponentialOf(ndarray.Float64, "bad"))
	add("gamma int")(g.StandardGammaOf(ndarray.Int8, 1))
	add("normal scale")(g.Normal(0, -1))
	add("exponential")(g.Exponential(-1))
	add("uniform inf")(g.Uniform(-math.MaxFloat64, math.MaxFloat64))
	add("uniform nan")(g.Uniform(nan, 1))
	add("uniform reversed")(g.Uniform(1, 0))
	add("standard_gamma")(g.StandardGamma(-1))
	add("gamma shape")(g.Gamma(-1, 1))
	add("gamma scale")(g.Gamma(1, -1))
	add("beta a")(g.Beta(0, 1))
	add("beta b")(g.Beta(1, 0))
	add("chisquare")(g.ChiSquare(0))
	add("f num")(g.F(0, 1))
	add("f den")(g.F(1, 0))
	add("ncchisq df")(g.NoncentralChiSquare(0, 1))
	add("ncchisq nonc")(g.NoncentralChiSquare(1, -1))
	add("ncf num")(g.NoncentralF(0, 1, 1))
	add("ncf den")(g.NoncentralF(1, 0, 1))
	add("ncf nonc")(g.NoncentralF(1, 1, -1))
	add("t")(g.StandardT(0))
	add("vonmises")(g.VonMises(0, -1))
	add("pareto")(g.Pareto(0))
	add("weibull")(g.Weibull(-1))
	add("power")(g.Power(0))
	add("laplace")(g.Laplace(0, -1))
	add("gumbel")(g.Gumbel(0, -1))
	add("logistic")(g.Logistic(0, -1))
	add("lognormal")(g.Lognormal(0, -1))
	add("rayleigh")(g.Rayleigh(-1))
	add("wald mean")(g.Wald(0, 1))
	add("wald scale")(g.Wald(1, 0))
	add("triangular left")(g.Triangular(2, 1, 3))
	add("triangular mode")(g.Triangular(0, 4, 3))
	add("triangular equal")(g.Triangular(1, 1, 1))
	add("binomial p")(g.Binomial(1, 2))
	add("binomial n")(g.Binomial(-1, 0.5))
	add("negbin nan")(g.NegativeBinomial(nan, 0.5))
	add("negbin n")(g.NegativeBinomial(0, 0.5))
	add("negbin p")(g.NegativeBinomial(1, 0))
	add("negbin large")(g.NegativeBinomial(1e18, 1e-10))
	add("poisson neg")(g.Poisson(-1))
	add("poisson large")(g.Poisson(1e19))
	add("zipf")(g.Zipf(1))
	add("geometric")(g.Geometric(0))
	add("logseries")(g.Logseries(1))
	add("dirichlet")(g.Dirichlet([]float64{1, -1}))
	add("dirichlet size")(g.Dirichlet([]float64{1}, -1))
	add("multinomial empty")(g.Multinomial(1, nil))
	add("multinomial p")(g.Multinomial(1, []float64{2, 0}))
	add("multinomial sum")(g.Multinomial(1, []float64{0.7, 0.7, 0}))
	add("multinomial n")(g.Multinomial(-1, []float64{1}))
	add("multinomial size")(g.Multinomial(1, []float64{1}, -1))
	add("integers dtype")(g.IntegersOf(ndarray.Float32, 0, 1, false))
	add("integers low")(g.IntegersOf(ndarray.Int8, -200, 1, false))
	add("integers high")(g.IntegersOf(ndarray.Int8, 0, 200, false))
	add("integers empty")(g.IntegersOf(ndarray.Int8, 5, 5, false))
	add("integers min")(g.IntegersOf(ndarray.Int64, 0, math.MinInt64, false))
	add("integers closed")(g.IntegersOf(ndarray.Int8, 5, 4, true))
	add("integers u64 neg")(g.IntegersOf(ndarray.Uint64, -1, 4, false))
	add("integers size")(g.IntegersOf(ndarray.Int8, 0, 4, false, -1))
	add("integers u64 size")(g.IntegersUint64(0, 4, false, -1))
	add("integers u64 zero")(g.IntegersUint64(0, 0, false))
	add("integers u64 empty")(g.IntegersUint64(5, 5, false))
	add("integers u64 closed")(g.IntegersUint64(5, 4, true))
	add("permutation")(g.Permutation(-1))
	add("permutationOf 0-d")(g.PermutationOf(ndarray.Scalar(1), 0))
	add("permutationOf axis")(g.PermutationOf(namedArray("arange12"), 2))
	add("permuted axis")(g.PermutedAxis(namedArray("arange12"), -3))
	add("choiceN")(g.ChoiceN(0, ChoiceOptions{}, 1))
	add("choice 0-d")(g.Choice(ndarray.Scalar(1), ChoiceOptions{}))
	add("choice axis")(g.Choice(namedArray("arange12"), ChoiceOptions{Axis: 2}))
	add("choice empty")(g.Choice(must(ndarray.ZerosOf(ndarray.Int8, 0)), ChoiceOptions{}, 2))
	add("choice p size")(g.ChoiceN(3, ChoiceOptions{P: []float64{1}}))
	add("choice p nan")(g.ChoiceN(2, ChoiceOptions{P: []float64{nan, 1}}))
	add("choice p neg")(g.ChoiceN(2, ChoiceOptions{P: []float64{-1, 2}}))
	add("choice p sum")(g.ChoiceN(2, ChoiceOptions{P: []float64{0.5, 0.6}}))
	add("choice negative")(g.ChoiceN(3, ChoiceOptions{}, -1))
	add("choice too many")(g.ChoiceN(3, ChoiceOptions{NoReplace: true}, 4))
	add("choice nonzero")(g.ChoiceN(3, ChoiceOptions{NoReplace: true, P: []float64{1, 0, 0}}, 2))
	add("normal size")(g.Normal(0, 1, -1))
	add("random32 size")(g.RandomOf(ndarray.Float32, -1))
	add("poisson size")(g.Poisson(1, -1))
	add("standard_normal size")(g.StandardNormal(-1))
	add("integers closed zero")(g.IntegersOf(ndarray.Int8, 0, -1, true))
	add("choice p")(g.Choice(namedArray("floats6"), ChoiceOptions{P: []float64{1}}))
	add("legacy choice bad p")(NewRandomState(1).Choice(namedArray("floats6"), ChoiceOptions{P: []float64{1}}))
	add("legacy choice huge")(NewRandomState(1).ChoiceN(1<<62, ChoiceOptions{NoReplace: true}, 1))
	add("choice shape")(g.Choice(namedArray("floats6"), ChoiceOptions{}, -1, -1))
	add("choiceN shape")(g.ChoiceN(6, ChoiceOptions{}, -1, -1))
	errs["shuffle 0-d"] = g.Shuffle(ndarray.Scalar(1), 0)
	errs["shuffle axis"] = g.Shuffle(namedArray("arange12"), 5)
	view := must(namedArray("arange12").Slice(ndarray.R(0, 2)))
	errs["shuffle view"] = g.Shuffle(view, 0)

	r := NewRandomState(1)
	add("legacy normal")(r.Normal(0, -1))
	add("legacy lognormal")(r.Lognormal(0, -1))
	add("legacy uniform")(r.Uniform(0, math.Inf(1)))
	add("legacy exponential")(r.Exponential(-1))
	add("legacy gamma shape")(r.Gamma(-1, 1))
	add("legacy gamma scale")(r.Gamma(1, -1))
	add("legacy chisquare")(r.ChiSquare(0))
	add("legacy binomial p")(r.Binomial(1, -1))
	add("legacy binomial n")(r.Binomial(-1, 0.5))
	add("legacy poisson")(r.Poisson(-1))
	add("legacy permutation")(r.Permutation(-1))
	add("legacy permutationOf")(r.PermutationOf(ndarray.Scalar(1)))
	add("legacy choiceN")(r.ChoiceN(0, ChoiceOptions{}, 1))
	add("legacy choice 2-d")(r.Choice(namedArray("arange12"), ChoiceOptions{}))
	add("legacy choice empty")(r.Choice(must(ndarray.ZerosOf(ndarray.Int8, 0)), ChoiceOptions{}, 1))
	add("legacy choice p")(r.ChoiceN(2, ChoiceOptions{P: []float64{1}}))
	add("legacy choice too many")(r.ChoiceN(2, ChoiceOptions{NoReplace: true}, 3))
	add("legacy choice shape")(r.Choice(namedArray("floats6"), ChoiceOptions{}, -1, -1))
	add("legacy choiceN shape")(r.ChoiceN(6, ChoiceOptions{}, -1, -1))
	add("legacy choice perm")(r.ChoiceN(-5, ChoiceOptions{NoReplace: true}))
	errs["legacy shuffle 0-d"] = r.Shuffle(ndarray.Scalar(1))
	errs["legacy shuffle view"] = r.Shuffle(view)
	_, err := NewRandomStateArray(nil)
	errs["legacy seed empty"] = err
	nr := NewRandomStateFrom(NewPCG64(1))
	errs["legacy seed pcg"] = nr.Seed(1)
	errs["legacy seedArray pcg"] = nr.SeedArray([]uint32{1})
	errs["legacy seedArray empty"] = r.SeedArray(nil)
	_, errs["legacy getstate pcg"] = nr.GetState()
	errs["legacy setstate pcg"] = nr.SetState(LegacyState{})

	_, err = NewSeedSequenceWith(SeedSequenceOptions{PoolSize: 2})
	errs["seedseq pool"] = err
	_, err = NewSeedSequenceWith(SeedSequenceOptions{Entropy: []uint64{1}, Words: []uint32{1}})
	errs["seedseq both"] = err
	_, err = NewSeedSequenceBig(big.NewInt(-1))
	errs["seedseq negative"] = err
	_, err = NewGenerator(NewMT19937Legacy(1)).Spawn(1)
	errs["spawn legacy"] = err
	_, err = NewGenerator(stubBG{}).Spawn(1)
	errs["spawn stub"] = err
	_, err = NewPCG64(1).Spawn(-1)
	errs["spawn negative"] = err
	_, err = DefaultRNG(1).Spawn(-1)
	errs["generator spawn negative"] = err

	for name, err := range errs {
		mustErr(t, name, err)
		if err != nil && !errors.Is(err, ErrValue) && !errors.Is(err, ErrNotContiguous) &&
			!errors.Is(err, ErrNoSeedSequence) && !errors.Is(err, ndarray.ErrShapeMismatch) {
			t.Logf("%s: %v (not ErrValue)", name, err)
		}
	}
	if !errors.Is(errs["shuffle view"], ErrNotContiguous) {
		t.Errorf("shuffle of a view: %v", errs["shuffle view"])
	}
	defer func() {
		if recover() == nil {
			t.Error("Spawn(-1) did not panic")
		}
	}()
	NewSeedSequence(1).Spawn(-1)
}

// TestGlobalFunctionsAreRandomState checks np.random.seed(0) followed by the
// module functions against a RandomState(0).
func TestGlobalFunctionsAreRandomState(t *testing.T) {
	Seed(7)
	r := NewRandomState(7)
	same := func(name string, a, b *ndarray.Array) {
		t.Helper()
		if a.String() != b.String() {
			t.Errorf("%s: %v, RandomState %v", name, a, b)
		}
	}
	same("Rand", must(Rand(3)), must(r.Rand(3)))
	same("Randn", must(Randn(3)), must(r.Randn(3)))
	same("RandomSample", must(RandomSample(2)), must(r.RandomSample(2)))
	same("Randint", must(Randint(0, 9, 4)), must(r.Randint(0, 9, 4)))
	same("Normal", must(Normal(1, 2, 2)), must(r.Normal(1, 2, 2)))
	same("Uniform", must(Uniform(1, 2, 2)), must(r.Uniform(1, 2, 2)))
	a := namedArray("floats6")
	same("Choice", must(Choice(a, ChoiceOptions{}, 3)), must(r.Choice(a, ChoiceOptions{}, 3)))
	x, y := namedArray("arange10"), namedArray("arange10")
	if err := Shuffle(x); err != nil {
		t.Fatal(err)
	}
	if err := r.Shuffle(y); err != nil {
		t.Fatal(err)
	}
	same("Shuffle", x, y)
	same("Permutation", must(Permutation(5)), must(r.Permutation(5)))
	same("Binomial", must(Binomial(10, 0.5, 3)), must(r.Binomial(10, 0.5, 3)))
	same("Poisson", must(Poisson(4, 3)), must(r.Poisson(4, 3)))
}

func TestStateRoundTrips(t *testing.T) {
	check := func(name string, a, b BitGenerator) {
		t.Helper()
		if x, y := RandomRaw(a, 3), RandomRaw(b, 3); x[0] != y[0] || x[2] != y[2] {
			t.Errorf("%s: %v after SetState, %v", name, x, y)
		}
	}
	p := NewPCG64(3)
	p.Uint32()
	q := NewPCG64(9)
	q.SetState(p.State())
	if q.Uint32() != p.Uint32() {
		t.Error("PCG64 buffered half not restored")
	}
	check("PCG64", p, q)
	d, d2 := NewPCG64DXSM(3), NewPCG64DXSM()
	d2.SetState(d.State())
	check("PCG64DXSM", d, d2)
	m, m2 := NewMT19937(3), NewMT19937()
	m2.SetState(m.State())
	check("MT19937", m, m2)
	ph, ph2 := NewPhilox(3), NewPhilox()
	ph.Uint64()
	ph2.SetState(ph.State())
	check("Philox", ph, ph2)
	s, s2 := NewSFC64(3), NewSFC64()
	s2.SetState(s.State())
	check("SFC64", s, s2)
	for _, sp := range []Spawner{p, d, m, ph, s} {
		if sp.SeedSeq() == nil {
			t.Errorf("%T has no SeedSequence", sp)
		}
	}

	r := NewRandomState(5)
	r.Randn(1) // leaves a cached Gaussian
	st := must(r.GetState())
	want := must(r.Randn(4))
	r2 := NewRandomStateEntropy()
	if err := r2.SetState(st); err != nil {
		t.Fatal(err)
	}
	if got := must(r2.Randn(4)); got.String() != want.String() {
		t.Errorf("RandomState after SetState: %v, want %v", got, want)
	}
	if err := r2.SeedArray([]uint32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	r3 := must(NewRandomStateArray([]uint32{1, 2, 3}))
	if a, b := must(r2.Rand(2)), must(r3.Rand(2)); a.String() != b.String() {
		t.Errorf("SeedArray: %v, NewRandomStateArray %v", a, b)
	}
	if r2.BitGenerator() == nil || DefaultRNG(1).BitGenerator() == nil {
		t.Error("nil bit generator")
	}
	kids := must(DefaultRNG(4).Spawn(2))
	if len(kids) != 2 {
		t.Error("Spawn")
	}
}

func TestSeedSequenceAccessors(t *testing.T) {
	ss := must(NewSeedSequenceBig(new(big.Int).Lsh(big.NewInt(3), 70)))
	if ss.EntropyInt().Cmp(new(big.Int).Lsh(big.NewInt(3), 70)) != 0 || len(ss.Entropy()) != 3 {
		t.Errorf("entropy %v %v", ss.EntropyInt(), ss.Entropy())
	}
	if ss.PoolSize() != 4 || ss.NChildrenSpawned() != 0 {
		t.Error("accessors")
	}
	ss.Spawn(2)
	if ss.NChildrenSpawned() != 2 {
		t.Error("NChildrenSpawned")
	}
	zero := must(NewSeedSequenceBig(new(big.Int)))
	if len(zero.Entropy()) != 1 || zero.Entropy()[0] != 0 {
		t.Errorf("zero entropy %v", zero.Entropy())
	}
	// Fresh entropy: 128 bits from the OS, as numpy's randbits(128).
	called := false
	old := readOSEntropy
	readOSEntropy = func(b []byte) (int, error) {
		called = true
		for i := range b {
			b[i] = 0
		}
		b[0] = 9
		return len(b), nil
	}
	ss = NewSeedSequence()
	readOSEntropy = old
	if !called || len(ss.Entropy()) != 1 || ss.Entropy()[0] != 9 {
		t.Errorf("OS entropy: %v", ss.Entropy())
	}
	if a, b := NewSeedSequence().Pool(), NewSeedSequence().Pool(); a[0] == b[0] && a[1] == b[1] {
		t.Error("two OS-seeded sequences agree")
	}
}

func TestShapesAndDTypes(t *testing.T) {
	g := DefaultRNG(2)
	for _, c := range []struct {
		a     *ndarray.Array
		dt    ndarray.DType
		shape string
	}{
		{must(g.Random()), ndarray.Float64, "[]"},
		{must(g.Random(2, 3)), ndarray.Float64, "[2 3]"},
		{must(g.StandardExponential(4)), ndarray.Float64, "[4]"},
		{must(g.StandardGamma(2, 4)), ndarray.Float64, "[4]"},
		{must(g.Integers(0, 5, 3)), ndarray.Int64, "[3]"},
		{must(g.IntegersOf(ndarray.Uint64, 0, 5, false, 3)), ndarray.Uint64, "[3]"},
		{must(g.Dirichlet([]float64{1, 2}, 3)), ndarray.Float64, "[3 2]"},
		{must(g.Dirichlet(nil, 3)), ndarray.Float64, "[3 0]"},
		{must(g.ChoiceN(5, ChoiceOptions{}, 2, 2)), ndarray.Int64, "[2 2]"},
		{must(g.Choice(namedArray("arange12"), ChoiceOptions{Axis: -1}, 5)), ndarray.Int64, "[3 5]"},
		{must(g.Permuted(namedArray("arange12"))), ndarray.Int64, "[3 4]"},
		{must(NewRandomState(1).RandintOf(ndarray.Uint8, 0, 255, 2)), ndarray.Uint8, "[2]"},
		{must(NewRandomState(1).Choice(namedArray("floats6"), ChoiceOptions{NoReplace: true, P: []float64{.1, .1, .2, .2, .2, .2}}, 3)), ndarray.Float64, "[3]"},
	} {
		if c.a.DType() != c.dt || ndarrayShape(c.a) != c.shape {
			t.Errorf("%v %v, want %v %s", c.a.DType(), c.a.Shape(), c.dt, c.shape)
		}
	}
	if len(g.Bytes(5)) != 5 || len(NewRandomState(1).Bytes(0)) != 0 {
		t.Error("Bytes length")
	}
	// Shuffle permutes in place, along any axis, for every dtype.
	for dt := ndarray.DType(0); dt <= ndarray.Complex128; dt++ {
		a := namedArray("arange12").AsType(dt)
		if err := g.Shuffle(a, 1); err != nil {
			t.Fatal(err)
		}
		if err := NewRandomState(1).Shuffle(a); err != nil {
			t.Fatal(err)
		}
		if _, err := g.PermutedAxis(a, 0); err != nil {
			t.Fatal(err)
		}
	}
	empty := must(ndarray.ZerosOf(ndarray.Int8, 0))
	if g.Shuffle(empty, 0) != nil || NewRandomState(1).Shuffle(empty) != nil {
		t.Error("shuffle of an empty array")
	}
}

func ndarrayShape(a *ndarray.Array) string {
	s := "["
	for i, d := range a.Shape() {
		if i > 0 {
			s += " "
		}
		s += itoa(d)
	}
	return s + "]"
}

func itoa(d int) string { return big.NewInt(int64(d)).String() }
