package random

import (
	"encoding/binary"
	"math"
	"reflect"
	"sync"

	"github.com/go-ndarray/ndarray"
)

// RandomState is numpy's legacy numpy.random.RandomState: MT19937 with
// NumPy 1.16's samplers, frozen so that old seeds keep their streams. It is
// what np.random.seed(0) followed by np.random.rand(...) uses, and it
// reproduces those numbers bit for bit.
//
// A RandomState is not safe for concurrent use; the package-level functions
// (Seed, Rand, Randn ...) share one behind a mutex, as numpy's module-level
// functions share one behind a lock.
type RandomState struct {
	bg       BitGenerator
	hasGauss bool
	gauss    float64
	binom    binomialCache
}

// NewRandomState returns numpy's RandomState(seed) for an integer seed.
func NewRandomState(seed uint32) *RandomState {
	return &RandomState{bg: NewMT19937Legacy(seed)}
}

// NewRandomStateArray returns numpy's RandomState(seed) for an array seed
// (init_by_array).
func NewRandomStateArray(seed []uint32) (*RandomState, error) {
	m := &MT19937{}
	if err := m.SeedLegacyArray(seed); err != nil {
		return nil, err
	}
	return &RandomState{bg: m}, nil
}

// NewRandomStateFrom returns numpy's RandomState(bit_generator): the legacy
// samplers over any bit generator.
func NewRandomStateFrom(bg BitGenerator) *RandomState { return &RandomState{bg: bg} }

// NewRandomStateEntropy returns numpy's RandomState() (seed=None): MT19937
// seeded from a fresh SeedSequence.
func NewRandomStateEntropy() *RandomState { return &RandomState{bg: NewMT19937()} }

// BitGenerator returns the underlying bit generator.
func (r *RandomState) BitGenerator() BitGenerator { return r.bg }

func (r *RandomState) mt() (*MT19937, error) {
	m, ok := r.bg.(*MT19937)
	if !ok {
		return nil, valueErr("can only re-seed a MT19937 BitGenerator")
	}
	return m, nil
}

// Seed reseeds as np.random.seed(seed) with an integer seed and drops a
// cached Gaussian.
func (r *RandomState) Seed(seed uint32) error {
	m, err := r.mt()
	if err != nil {
		return err
	}
	m.SeedLegacy(seed)
	r.hasGauss, r.gauss = false, 0
	return nil
}

// SeedArray reseeds as np.random.seed(seed) with an array seed.
func (r *RandomState) SeedArray(seed []uint32) error {
	m, err := r.mt()
	if err != nil {
		return err
	}
	if err := m.SeedLegacyArray(seed); err != nil {
		return err
	}
	r.hasGauss, r.gauss = false, 0
	return nil
}

// LegacyState is numpy's legacy get_state() tuple: the MT19937 key and
// position and the cached Gaussian.
type LegacyState struct {
	MT19937State
	HasGauss bool
	Gauss    float64
}

// GetState returns the state, numpy's get_state() for an MT19937.
func (r *RandomState) GetState() (LegacyState, error) {
	m, err := r.mt()
	if err != nil {
		return LegacyState{}, err
	}
	return LegacyState{m.State(), r.hasGauss, r.gauss}, nil
}

// SetState restores a state from GetState, numpy's set_state.
func (r *RandomState) SetState(s LegacyState) error {
	m, err := r.mt()
	if err != nil {
		return err
	}
	m.SetState(s.MT19937State)
	r.hasGauss, r.gauss = s.HasGauss, s.Gauss
	return nil
}

// gaussian is numpy's legacy_gauss: the polar method, which makes two values
// and keeps the second for the next call.
func (r *RandomState) gaussian() float64 {
	if r.hasGauss {
		t := r.gauss
		r.hasGauss, r.gauss = false, 0
		return t
	}
	var x1, x2, r2 float64
	for {
		x1 = 2.0*r.bg.Float64() - 1.0
		x2 = 2.0*r.bg.Float64() - 1.0
		r2 = float64(x1*x1) + float64(x2*x2)
		if r2 < 1.0 && r2 != 0.0 {
			break
		}
	}
	f := math.Sqrt(-2.0 * log(r2) / r2)
	r.gauss = f * x1
	r.hasGauss = true
	return f * x2
}

func (r *RandomState) standardExponential() float64 { return -log(1.0 - r.bg.Float64()) }

// standardGamma is numpy's legacy_standard_gamma: Generator's algorithm over
// the legacy Gaussian and exponential.
func (r *RandomState) standardGamma(shape float64) float64 {
	switch {
	case shape == 1.0:
		return r.standardExponential()
	case shape == 0.0:
		return 0.0
	case shape < 1.0:
		for {
			u := r.bg.Float64()
			v := r.standardExponential()
			if u <= 1.0-shape {
				x := pow(u, 1./shape)
				if x <= v {
					return x
				}
			} else {
				y := -log((1 - u) / shape)
				x := pow(1.0-shape+float64(shape*y), 1./shape)
				if x <= v+y {
					return x
				}
			}
		}
	}
	b := shape - 1./3.
	c := 1. / math.Sqrt(9*b)
	for {
		var x, v float64
		for {
			x = r.gaussian()
			v = 1.0 + float64(c*x)
			if v > 0.0 {
				break
			}
		}
		v = v * v * v
		u := r.bg.Float64()
		if u < 1.0-float64(0.0331*(x*x)*(x*x)) {
			return b * v
		}
		if log(u) < float64(0.5*x*x)+float64(b*(1.-v+log(v))) {
			return b * v
		}
	}
}

// RandomSample returns float64 values in [0, 1), numpy's random_sample(size).
func (r *RandomState) RandomSample(size ...int) (*ndarray.Array, error) {
	return fill64(size, r.bg.Float64)
}

// Rand is numpy's rand(d0, d1, ...): RandomSample with the dimensions as
// arguments.
func (r *RandomState) Rand(dims ...int) (*ndarray.Array, error) { return r.RandomSample(dims...) }

// Randn is numpy's randn(d0, d1, ...): standard normal values from the
// legacy polar method.
func (r *RandomState) Randn(dims ...int) (*ndarray.Array, error) { return r.StandardNormal(dims...) }

// StandardNormal is numpy's legacy standard_normal(size).
func (r *RandomState) StandardNormal(size ...int) (*ndarray.Array, error) {
	return fill64(size, r.gaussian)
}

// Normal is numpy's legacy normal(loc, scale, size).
func (r *RandomState) Normal(loc, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return loc + float64(scale*r.gaussian()) })
}

// Lognormal is numpy's legacy lognormal(mean, sigma, size).
func (r *RandomState) Lognormal(mean, sigma float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(sigma, "sigma"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return exp(mean + float64(sigma*r.gaussian())) })
}

// Uniform is numpy's legacy uniform(low, high, size); unlike Generator's it
// accepts high < low.
func (r *RandomState) Uniform(low, high float64, size ...int) (*ndarray.Array, error) {
	rng := high - low
	if math.IsInf(rng, 0) || math.IsNaN(rng) {
		return nil, valueErr("Range exceeds valid bounds")
	}
	return fill64(size, func() float64 { return uniform(r.bg, low, rng) })
}

// StandardExponential is numpy's legacy standard_exponential(size): -log(1-U).
func (r *RandomState) StandardExponential(size ...int) (*ndarray.Array, error) {
	return fill64(size, r.standardExponential)
}

// Exponential is numpy's legacy exponential(scale, size).
func (r *RandomState) Exponential(scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return scale * r.standardExponential() })
}

// StandardGamma is numpy's legacy standard_gamma(shape, size).
func (r *RandomState) StandardGamma(shape float64, size ...int) (*ndarray.Array, error) {
	return r.Gamma(shape, 1, size...)
}

// Gamma is numpy's legacy gamma(shape, scale, size).
func (r *RandomState) Gamma(shape, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(shape, "shape"); err != nil {
		return nil, err
	}
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return scale * r.standardGamma(shape) })
}

// ChiSquare is numpy's legacy chisquare(df, size).
func (r *RandomState) ChiSquare(df float64, size ...int) (*ndarray.Array, error) {
	if err := positive(df, "df"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return 2.0 * r.standardGamma(df/2.0) })
}

// Randint returns int64 values in [low, high), numpy's randint(low, high,
// size) with its default dtype (C long, which is int64 on Linux and macOS
// but int32 on Windows; this always uses int64), by masked rejection.
func (r *RandomState) Randint(low, high int64, size ...int) (*ndarray.Array, error) {
	return r.RandintOf(ndarray.Int64, low, high, size...)
}

// RandintOf is numpy's randint(low, high, size, dtype).
func (r *RandomState) RandintOf(dt ndarray.DType, low, high int64, size ...int) (*ndarray.Array, error) {
	return integers(r.bg, true, dt, low, high, false, size)
}

// Bytes returns n random bytes, numpy's legacy bytes(n).
func (r *RandomState) Bytes(n int) []byte {
	words := (n + 3) / 4
	b := make([]byte, 4*words)
	for i := range words {
		binary.LittleEndian.PutUint32(b[4*i:], r.bg.Uint32())
	}
	return b[:n]
}

// Binomial is numpy's legacy binomial(n, p, size), int64 values.
func (r *RandomState) Binomial(n int64, p float64, size ...int) (*ndarray.Array, error) {
	if err := bounded01(p, "p"); err != nil {
		return nil, err
	}
	if n < 0 {
		return nil, valueErr("n < 0")
	}
	return fillInt(size, func() int64 { return binomial(r.bg, p, n, &r.binom, true) })
}

// Poisson is numpy's legacy poisson(lam, size), int64 values.
func (r *RandomState) Poisson(lam float64, size ...int) (*ndarray.Array, error) {
	if err := poissonLam(lam); err != nil {
		return nil, err
	}
	return fillInt(size, func() int64 { return poisson(r.bg, lam) })
}

// Multinomial is numpy's legacy multinomial(n, pvals, size).
func (r *RandomState) Multinomial(n int64, pvals []float64, size ...int) (*ndarray.Array, error) {
	return multinomialArray(r.bg, &r.binom, true, n, pvals, size)
}

// Shuffle permutes a in place along its first axis, numpy's legacy
// shuffle(x). a must own contiguous storage.
func (r *RandomState) Shuffle(a *ndarray.Array) error {
	if a.Ndim() == 0 {
		return valueErr("len() of unsized object")
	}
	s, err := ownStorage(a)
	if err != nil || a.Size() == 0 {
		return err
	}
	outer, n, inner := layout(a.Shape(), 0)
	swap := reflect.Swapper(s.Interface())
	for i := n - 1; i >= 1; i-- {
		j := int(interval(r.bg, uint64(i)))
		if i != j {
			swapSlabs(swap, outer, n, inner, i, j)
		}
	}
	return nil
}

// Permutation returns a shuffled arange(n) of int64, numpy's legacy
// permutation(n).
func (r *RandomState) Permutation(n int) (*ndarray.Array, error) {
	a, d, err := newArray[int64](ndarray.Int64, []int{n})
	if err != nil {
		return nil, err
	}
	for i := range d {
		d[i] = int64(i)
	}
	return a, r.Shuffle(a)
}

// PermutationOf returns a copy of a with its first axis shuffled, numpy's
// legacy permutation(x).
func (r *RandomState) PermutationOf(a *ndarray.Array) (*ndarray.Array, error) {
	if a.Ndim() == 0 {
		return nil, valueErr("x must be an integer or at least 1-dimensional")
	}
	if a.Ndim() == 1 {
		c := a.Copy()
		return c, r.Shuffle(c)
	}
	idx, _ := r.Permutation(a.Shape()[0])
	d, _ := ndarray.Data[int64](idx)
	return takeAxis(a, d, []int{len(d)}, 0)
}

// ChoiceN is numpy's legacy choice(n, size, replace, p); o.Axis and
// o.NoShuffle do not apply.
func (r *RandomState) ChoiceN(n int64, o ChoiceOptions, size ...int) (*ndarray.Array, error) {
	if n <= 0 && ndarrayProd(size) != 0 {
		return nil, valueErr("a must be greater than 0 unless no samples are taken")
	}
	idx, err := r.choiceIndices(n, o, size)
	if err != nil {
		return nil, err
	}
	return ndarray.FromSlice(idx, size...)
}

// Choice is numpy's legacy choice(a, size, replace, p) for a 1-d a.
func (r *RandomState) Choice(a *ndarray.Array, o ChoiceOptions, size ...int) (*ndarray.Array, error) {
	if a.Ndim() != 1 {
		return nil, valueErr("a must be 1-dimensional")
	}
	n := int64(a.Shape()[0])
	if n == 0 && ndarrayProd(size) != 0 {
		return nil, valueErr("'a' cannot be empty unless no samples are taken")
	}
	idx, err := r.choiceIndices(n, o, size)
	if err != nil {
		return nil, err
	}
	return takeAxis(a, idx, size, 0)
}

func (r *RandomState) choiceIndices(n int64, o ChoiceOptions, size []int) ([]int64, error) {
	if o.P != nil {
		if err := checkP(o.P, n); err != nil {
			return nil, err
		}
	}
	count := ndarrayProd(size)
	idx := make([]int64, count)
	// The weighted paths draw exactly as Generator's do (random_sample and
	// searchsorted); only the unweighted ones differ.
	g := &Generator{bg: r.bg}
	switch {
	case !o.NoReplace && o.P != nil:
		return g.choiceIndices(n, o, size)
	case !o.NoReplace:
		fillUint64(r.bg, 0, uint64(n-1), true, asUint64(idx))
		return idx, nil
	case int64(count) > n:
		return nil, valueErr("Cannot take a larger sample than population when 'replace=False'")
	case o.P != nil:
		return g.choiceWeighted(o.P, idx)
	}
	// Beyond int (32-bit platforms) the allocation fails as too big.
	perm, err := r.Permutation(int(min(n, int64(math.MaxInt))))
	if err != nil {
		return nil, err
	}
	d, _ := ndarray.Data[int64](perm)
	copy(idx, d)
	return idx, nil
}

// global is the RandomState behind the package-level functions, numpy's
// np.random._rand, seeded from the OS until Seed is called.
var (
	globalMu sync.Mutex
	global   *RandomState
)

func globalState() *RandomState {
	if global == nil {
		global = NewRandomStateEntropy()
	}
	return global
}

func withGlobal[T any](f func(r *RandomState) T) T {
	globalMu.Lock()
	defer globalMu.Unlock()
	return f(globalState())
}

// Seed is np.random.seed(seed): it reseeds the shared RandomState.
func Seed(seed uint32) {
	withGlobal(func(r *RandomState) error { return r.Seed(seed) })
}

// Rand is np.random.rand(d0, d1, ...).
func Rand(dims ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Rand(dims...)) }).unpack()
}

// Randn is np.random.randn(d0, d1, ...).
func Randn(dims ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Randn(dims...)) }).unpack()
}

// RandomSample is np.random.random_sample(size) (also np.random.random).
func RandomSample(size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.RandomSample(size...)) }).unpack()
}

// Randint is np.random.randint(low, high, size).
func Randint(low, high int64, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Randint(low, high, size...)) }).unpack()
}

// Normal is np.random.normal(loc, scale, size).
func Normal(loc, scale float64, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Normal(loc, scale, size...)) }).unpack()
}

// Uniform is np.random.uniform(low, high, size).
func Uniform(low, high float64, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Uniform(low, high, size...)) }).unpack()
}

// Choice is np.random.choice(a, size, replace, p).
func Choice(a *ndarray.Array, o ChoiceOptions, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Choice(a, o, size...)) }).unpack()
}

// Shuffle is np.random.shuffle(x).
func Shuffle(a *ndarray.Array) error {
	return withGlobal(func(r *RandomState) error { return r.Shuffle(a) })
}

// Permutation is np.random.permutation(n).
func Permutation(n int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Permutation(n)) }).unpack()
}

// Binomial is np.random.binomial(n, p, size).
func Binomial(n int64, p float64, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Binomial(n, p, size...)) }).unpack()
}

// Poisson is np.random.poisson(lam, size).
func Poisson(lam float64, size ...int) (*ndarray.Array, error) {
	return withGlobal(func(r *RandomState) result { return wrap(r.Poisson(lam, size...)) }).unpack()
}

type result struct {
	a   *ndarray.Array
	err error
}

func wrap(a *ndarray.Array, err error) result { return result{a, err} }

func (r result) unpack() (*ndarray.Array, error) { return r.a, r.err }
