package random

import (
	"encoding/binary"
	"math"
	"reflect"
	"sort"

	"github.com/go-ndarray/ndarray"
)

// Generator draws from distributions, numpy.random.Generator. Every method
// takes the output shape last (numpy's size): no shape gives a 0-d array,
// which holds what numpy returns as a scalar for size=None and consumes the
// same draws.
//
// A Generator is not safe for concurrent use; give each goroutine its own,
// from Spawn.
type Generator struct {
	bg    BitGenerator
	binom binomialCache
}

// NewGenerator returns numpy's Generator(bit_generator).
func NewGenerator(bg BitGenerator) *Generator { return &Generator{bg: bg} }

// DefaultRNG returns numpy's default_rng(seed): a Generator over
// PCG64(SeedSequence(seed)). One value is numpy's integer seed, several its
// list seed (default_rng([1, 2, 3])); no value seeds from the OS.
func DefaultRNG(seed ...uint64) *Generator { return NewGenerator(NewPCG64(seed...)) }

// BitGenerator returns the generator's bit generator.
func (g *Generator) BitGenerator() BitGenerator { return g.bg }

// Spawn returns n independent Generators, numpy's Generator.spawn: over
// bit generators of the same kind seeded from the SeedSequence's children.
func (g *Generator) Spawn(n int) ([]*Generator, error) {
	sp, ok := g.bg.(Spawner)
	if !ok {
		return nil, ErrNoSeedSequence
	}
	bgs, err := sp.Spawn(n)
	if err != nil {
		return nil, err
	}
	out := make([]*Generator, len(bgs))
	for i, b := range bgs {
		out[i] = NewGenerator(b)
	}
	return out, nil
}

// fill64 makes a float64 array of the given shape from n draws of f.
func fill64(shape []int, f func() float64) (*ndarray.Array, error) {
	a, d, err := newArray[float64](ndarray.Float64, shape)
	if err != nil {
		return nil, err
	}
	for i := range d {
		d[i] = f()
	}
	return a, nil
}

func fill32(shape []int, f func() float32) (*ndarray.Array, error) {
	a, d, err := newArray[float32](ndarray.Float32, shape)
	if err != nil {
		return nil, err
	}
	for i := range d {
		d[i] = f()
	}
	return a, nil
}

func fillInt(shape []int, f func() int64) (*ndarray.Array, error) {
	a, d, err := newArray[int64](ndarray.Int64, shape)
	if err != nil {
		return nil, err
	}
	for i := range d {
		d[i] = f()
	}
	return a, nil
}

func floatOnly(dt ndarray.DType, what string) error {
	if dt != ndarray.Float64 && dt != ndarray.Float32 {
		return valueErr("Unsupported dtype %v for %s", dt, what)
	}
	return nil
}

// Random returns float64 values in [0, 1), numpy's random(size).
func (g *Generator) Random(size ...int) (*ndarray.Array, error) {
	a, d, err := newArray[float64](ndarray.Float64, size)
	if err != nil {
		return nil, err
	}
	if p, ok := g.bg.(*PCG64); ok { // the default, called directly
		for i := range d {
			d[i] = uint64ToDouble(p.Uint64())
		}
		return a, nil
	}
	for i := range d {
		d[i] = g.bg.Float64()
	}
	return a, nil
}

// RandomOf returns values in [0, 1) of dtype Float64 or Float32, numpy's
// random(size, dtype). Float32 values have 24 random bits, from 32-bit draws.
func (g *Generator) RandomOf(dt ndarray.DType, size ...int) (*ndarray.Array, error) {
	if err := floatOnly(dt, "random"); err != nil {
		return nil, err
	}
	if dt == ndarray.Float64 {
		return g.Random(size...)
	}
	if p, ok := g.bg.(*PCG64); ok {
		a, d, err := newArray[float32](ndarray.Float32, size)
		if err != nil {
			return nil, err
		}
		for i := range d {
			d[i] = float32(p.Uint32()>>8) * (1.0 / 16777216.0)
		}
		return a, nil
	}
	return fill32(size, func() float32 { return nextFloat(g.bg) })
}

// StandardNormal returns float64 standard normal values (ziggurat), numpy's
// standard_normal(size).
func (g *Generator) StandardNormal(size ...int) (*ndarray.Array, error) {
	a, d, err := newArray[float64](ndarray.Float64, size)
	if err != nil {
		return nil, err
	}
	if p, ok := g.bg.(*PCG64); ok {
		for i := range d {
			d[i] = standardNormalPCG(p)
		}
		return a, nil
	}
	for i := range d {
		d[i] = standardNormal(g.bg)
	}
	return a, nil
}

// StandardNormalOf is numpy's standard_normal(size, dtype) for Float64 or
// Float32 (a separate 32-bit ziggurat).
func (g *Generator) StandardNormalOf(dt ndarray.DType, size ...int) (*ndarray.Array, error) {
	if err := floatOnly(dt, "standard_normal"); err != nil {
		return nil, err
	}
	if dt == ndarray.Float32 {
		return fill32(size, func() float32 { return standardNormalF(g.bg) })
	}
	return g.StandardNormal(size...)
}

// Normal returns loc + scale * standard normal, numpy's normal(loc, scale, size).
func (g *Generator) Normal(loc, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return normal(g.bg, loc, scale) })
}

// StandardExponential returns float64 values of the unit exponential
// (ziggurat), numpy's standard_exponential(size).
func (g *Generator) StandardExponential(size ...int) (*ndarray.Array, error) {
	return g.StandardExponentialOf(ndarray.Float64, "zig", size...)
}

// StandardExponentialOf is numpy's standard_exponential(size, dtype, method)
// with method "zig" (ziggurat) or "inv" (inversion, -log1p(-U)).
func (g *Generator) StandardExponentialOf(dt ndarray.DType, method string, size ...int) (*ndarray.Array, error) {
	if err := floatOnly(dt, "standard_exponential"); err != nil {
		return nil, err
	}
	if method != "zig" && method != "inv" {
		return nil, valueErr("method must be 'zig' or 'inv'")
	}
	inv := method == "inv"
	if dt == ndarray.Float32 {
		return fill32(size, func() float32 {
			if inv {
				return float32(-log1p(-float64(nextFloat(g.bg))))
			}
			return standardExponentialF(g.bg)
		})
	}
	return fill64(size, func() float64 {
		if inv {
			return -log1p(-g.bg.Float64())
		}
		return standardExponential(g.bg)
	})
}

// Exponential returns scale * the unit exponential, numpy's
// exponential(scale, size).
func (g *Generator) Exponential(scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return scale * standardExponential(g.bg) })
}

// Uniform returns values in [low, high), numpy's uniform(low, high, size):
// low + (high-low)*U.
func (g *Generator) Uniform(low, high float64, size ...int) (*ndarray.Array, error) {
	rng := high - low
	if math.IsInf(rng, 0) || math.IsNaN(rng) {
		return nil, valueErr("high - low range exceeds valid bounds")
	}
	if err := nonNegative(rng, "high - low"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return uniform(g.bg, low, rng) })
}

// StandardGamma is numpy's standard_gamma(shape, size) (Marsaglia-Tsang).
func (g *Generator) StandardGamma(shape float64, size ...int) (*ndarray.Array, error) {
	return g.StandardGammaOf(ndarray.Float64, shape, size...)
}

// StandardGammaOf is numpy's standard_gamma(shape, size, dtype).
func (g *Generator) StandardGammaOf(dt ndarray.DType, shape float64, size ...int) (*ndarray.Array, error) {
	if err := floatOnly(dt, "standard_gamma"); err != nil {
		return nil, err
	}
	if err := nonNegative(shape, "shape"); err != nil {
		return nil, err
	}
	if dt == ndarray.Float32 {
		s := float32(shape)
		return fill32(size, func() float32 { return standardGammaF(g.bg, s) })
	}
	return fill64(size, func() float64 { return standardGamma(g.bg, shape) })
}

// Gamma is numpy's gamma(shape, scale, size).
func (g *Generator) Gamma(shape, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(shape, "shape"); err != nil {
		return nil, err
	}
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return scale * standardGamma(g.bg, shape) })
}

// Beta is numpy's beta(a, b, size).
func (g *Generator) Beta(a, b float64, size ...int) (*ndarray.Array, error) {
	if err := positive(a, "a"); err != nil {
		return nil, err
	}
	if err := positive(b, "b"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return beta(g.bg, a, b) })
}

// ChiSquare is numpy's chisquare(df, size).
func (g *Generator) ChiSquare(df float64, size ...int) (*ndarray.Array, error) {
	if err := positive(df, "df"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return chisquare(g.bg, df) })
}

// F is numpy's f(dfnum, dfden, size).
func (g *Generator) F(dfnum, dfden float64, size ...int) (*ndarray.Array, error) {
	if err := positive(dfnum, "dfnum"); err != nil {
		return nil, err
	}
	if err := positive(dfden, "dfden"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return fDist(g.bg, dfnum, dfden) })
}

// NoncentralChiSquare is numpy's noncentral_chisquare(df, nonc, size).
func (g *Generator) NoncentralChiSquare(df, nonc float64, size ...int) (*ndarray.Array, error) {
	if err := positive(df, "df"); err != nil {
		return nil, err
	}
	if err := nonNegative(nonc, "nonc"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return noncentralChisquare(g.bg, df, nonc) })
}

// NoncentralF is numpy's noncentral_f(dfnum, dfden, nonc, size).
func (g *Generator) NoncentralF(dfnum, dfden, nonc float64, size ...int) (*ndarray.Array, error) {
	if err := positive(dfnum, "dfnum"); err != nil {
		return nil, err
	}
	if err := positive(dfden, "dfden"); err != nil {
		return nil, err
	}
	if err := nonNegative(nonc, "nonc"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return noncentralF(g.bg, dfnum, dfden, nonc) })
}

// StandardCauchy is numpy's standard_cauchy(size).
func (g *Generator) StandardCauchy(size ...int) (*ndarray.Array, error) {
	return fill64(size, func() float64 { return standardCauchy(g.bg) })
}

// StandardT is numpy's standard_t(df, size).
func (g *Generator) StandardT(df float64, size ...int) (*ndarray.Array, error) {
	if err := positive(df, "df"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return standardT(g.bg, df) })
}

// VonMises is numpy's vonmises(mu, kappa, size).
func (g *Generator) VonMises(mu, kappa float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(kappa, "kappa"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return vonmises(g.bg, mu, kappa) })
}

// Pareto is numpy's pareto(a, size).
func (g *Generator) Pareto(a float64, size ...int) (*ndarray.Array, error) {
	if err := positive(a, "a"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return pareto(g.bg, a) })
}

// Weibull is numpy's weibull(a, size).
func (g *Generator) Weibull(a float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(a, "a"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return weibull(g.bg, a) })
}

// Power is numpy's power(a, size).
func (g *Generator) Power(a float64, size ...int) (*ndarray.Array, error) {
	if err := positive(a, "a"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return power(g.bg, a) })
}

// Laplace is numpy's laplace(loc, scale, size).
func (g *Generator) Laplace(loc, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return laplace(g.bg, loc, scale) })
}

// Gumbel is numpy's gumbel(loc, scale, size).
func (g *Generator) Gumbel(loc, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return gumbel(g.bg, loc, scale) })
}

// Logistic is numpy's logistic(loc, scale, size).
func (g *Generator) Logistic(loc, scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return logistic(g.bg, loc, scale) })
}

// Lognormal is numpy's lognormal(mean, sigma, size).
func (g *Generator) Lognormal(mean, sigma float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(sigma, "sigma"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return lognormal(g.bg, mean, sigma) })
}

// Rayleigh is numpy's rayleigh(scale, size).
func (g *Generator) Rayleigh(scale float64, size ...int) (*ndarray.Array, error) {
	if err := nonNegative(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return rayleigh(g.bg, scale) })
}

// Wald is numpy's wald(mean, scale, size).
func (g *Generator) Wald(mean, scale float64, size ...int) (*ndarray.Array, error) {
	if err := positive(mean, "mean"); err != nil {
		return nil, err
	}
	if err := positive(scale, "scale"); err != nil {
		return nil, err
	}
	return fill64(size, func() float64 { return wald(g.bg, mean, scale) })
}

// Triangular is numpy's triangular(left, mode, right, size).
func (g *Generator) Triangular(left, mode, right float64, size ...int) (*ndarray.Array, error) {
	switch {
	case left > mode:
		return nil, valueErr("left > mode")
	case mode > right:
		return nil, valueErr("mode > right")
	case left == right:
		return nil, valueErr("left == right")
	}
	return fill64(size, func() float64 { return triangular(g.bg, left, mode, right) })
}

// Binomial is numpy's binomial(n, p, size), int64 values.
func (g *Generator) Binomial(n int64, p float64, size ...int) (*ndarray.Array, error) {
	if err := bounded01(p, "p"); err != nil {
		return nil, err
	}
	if n < 0 {
		return nil, valueErr("n < 0")
	}
	return fillInt(size, func() int64 { return binomial(g.bg, p, n, &g.binom, false) })
}

// NegativeBinomial is numpy's negative_binomial(n, p, size), int64 values.
func (g *Generator) NegativeBinomial(n, p float64, size ...int) (*ndarray.Array, error) {
	if math.IsNaN(n) {
		return nil, valueErr("n must not be NaN")
	}
	if err := positive(n, "n"); err != nil {
		return nil, err
	}
	if err := boundedGT01(p, "p"); err != nil {
		return nil, err
	}
	if (1-p)/p*(n+float64(10*math.Sqrt(n))) > poissonLamMax {
		return nil, valueErr("n too large or p too small, see Generator.negative_binomial Notes")
	}
	return fillInt(size, func() int64 { return negativeBinomial(g.bg, n, p) })
}

// poissonLamMax is numpy's POISSON_LAM_MAX.
var poissonLamMax = float64(math.MaxInt64) - float64(math.Sqrt(float64(math.MaxInt64))*10)

// Poisson is numpy's poisson(lam, size), int64 values.
func (g *Generator) Poisson(lam float64, size ...int) (*ndarray.Array, error) {
	if err := poissonLam(lam); err != nil {
		return nil, err
	}
	return fillInt(size, func() int64 { return poisson(g.bg, lam) })
}

// Zipf is numpy's zipf(a, size), int64 values.
func (g *Generator) Zipf(a float64, size ...int) (*ndarray.Array, error) {
	if !(a > 1) {
		return nil, valueErr("a <= 1 or a is NaN")
	}
	return fillInt(size, func() int64 { return zipf(g.bg, a) })
}

// Geometric is numpy's geometric(p, size), int64 values.
func (g *Generator) Geometric(p float64, size ...int) (*ndarray.Array, error) {
	if err := boundedGT01(p, "p"); err != nil {
		return nil, err
	}
	return fillInt(size, func() int64 { return geometric(g.bg, p) })
}

// Logseries is numpy's logseries(p, size), int64 values.
func (g *Generator) Logseries(p float64, size ...int) (*ndarray.Array, error) {
	if !(p >= 0) || !(p < 1) {
		return nil, valueErr("p < 0, p >= 1 or p is NaN")
	}
	return fillInt(size, func() int64 { return logseries(g.bg, p) })
}

// Bytes returns n random bytes, numpy's bytes(n): little-endian uint32 draws,
// the same bytes on every platform.
func (g *Generator) Bytes(n int) []byte {
	words := (n + 3) / 4
	b := make([]byte, 4*words)
	for i := range words {
		binary.LittleEndian.PutUint32(b[4*i:], g.bg.Uint32())
	}
	return b[:n]
}

// Dirichlet is numpy's dirichlet(alpha, size): shape size + (len(alpha),).
func (g *Generator) Dirichlet(alpha []float64, size ...int) (*ndarray.Array, error) {
	k := len(alpha)
	amax := math.Inf(-1)
	for _, a := range alpha {
		if a < 0 {
			return nil, valueErr("alpha < 0")
		}
		amax = max(amax, a)
	}
	shape := append(append([]int(nil), size...), k)
	out, d, err := newArray[float64](ndarray.Float64, shape)
	if err != nil {
		return nil, err
	}
	if k > 0 && amax < 0.1 {
		csum := make([]float64, k)
		acc := 0.0
		for j := k - 1; j >= 0; j-- {
			acc += alpha[j]
			csum[j] = acc
		}
		if acc > 0 {
			for i := 0; i < len(d); i += k {
				acc := 1.
				for j := 0; j < k-1; j++ {
					v := beta(g.bg, alpha[j], csum[j+1])
					d[i+j] = acc * v
					acc *= 1. - v
					if csum[j+1] == 0 {
						break
					}
				}
				d[i+k-1] = acc
			}
		}
		return out, nil
	}
	for i := 0; i < len(d); i += k {
		acc := 0.
		for j := range k {
			d[i+j] = standardGamma(g.bg, alpha[j])
			acc += d[i+j]
		}
		inv := 1. / acc
		for j := range k {
			d[i+j] *= inv
		}
	}
	return out, nil
}

// Multinomial is numpy's multinomial(n, pvals, size): int64 counts of shape
// size + (len(pvals),).
func (g *Generator) Multinomial(n int64, pvals []float64, size ...int) (*ndarray.Array, error) {
	return multinomialArray(g.bg, &g.binom, false, n, pvals, size)
}

func multinomialArray(bg BitGenerator, bc *binomialCache, legacy bool, n int64, pvals []float64, size []int) (*ndarray.Array, error) {
	d := len(pvals)
	if d == 0 {
		return nil, valueErr("pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.")
	}
	for _, p := range pvals {
		if err := bounded01(p, "pvals"); err != nil {
			return nil, err
		}
	}
	if kahanSum(pvals[:d-1]) > 1.0+1e-12 {
		return nil, valueErr("sum(pvals[:-1]) > 1.0")
	}
	if n < 0 {
		return nil, valueErr("n < 0")
	}
	shape := append(append([]int(nil), size...), d)
	out, m, err := newArray[int64](ndarray.Int64, shape)
	if err != nil {
		return nil, err
	}
	for off := 0; off < len(m); off += d {
		multinomial(bg, n, m[off:off+d], pvals, bc, legacy)
	}
	return out, nil
}

// kahanSum is numpy's kahan_sum.
func kahanSum(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	sum, c := x[0], 0.0
	for _, v := range x[1:] {
		y := v - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}
	return sum
}

func nonNegative(v float64, name string) error {
	if !math.IsNaN(v) && math.Signbit(v) {
		return valueErr("%s < 0", name)
	}
	return nil
}

func positive(v float64, name string) error {
	if v <= 0 {
		return valueErr("%s <= 0", name)
	}
	return nil
}

func bounded01(v float64, name string) error {
	if !(v >= 0) || !(v <= 1) {
		return valueErr("%s < 0, %s > 1 or %s is NaN", name, name, name)
	}
	return nil
}

func boundedGT01(v float64, name string) error {
	if !(v > 0) || !(v <= 1) {
		return valueErr("%s <= 0, %s > 1 or %s contains NaNs", name, name, name)
	}
	return nil
}

func poissonLam(lam float64) error {
	if !(lam >= 0) {
		return valueErr("lam < 0 or lam is NaN")
	}
	if !(lam <= poissonLamMax) {
		return valueErr("lam value too large")
	}
	return nil
}

// Shuffle permutes a in place along axis, numpy's shuffle(x, axis): the
// Fisher-Yates swaps of slabs along the axis, from the last to the second.
// a must own contiguous storage (not be a view).
func (g *Generator) Shuffle(a *ndarray.Array, axis int) error {
	if a.Ndim() == 0 {
		return valueErr("x must be an array of at least one dimension")
	}
	ax, err := normAxis(axis, a.Ndim())
	if err != nil {
		return err
	}
	s, err := ownStorage(a)
	if err != nil || a.Size() == 0 {
		return err
	}
	outer, n, inner := layout(a.Shape(), ax)
	swap := reflect.Swapper(s.Interface())
	for i := n - 1; i >= 1; i-- {
		j := int(interval(g.bg, uint64(i)))
		if i != j {
			swapSlabs(swap, outer, n, inner, i, j)
		}
	}
	return nil
}

// Permutation returns a shuffled arange(n) of int64, numpy's permutation(n).
func (g *Generator) Permutation(n int) (*ndarray.Array, error) {
	a, d, err := newArray[int64](ndarray.Int64, []int{n})
	if err != nil {
		return nil, err
	}
	for i := range d {
		d[i] = int64(i)
	}
	return a, g.Shuffle(a, 0)
}

// PermutationOf returns a copy of a with its slices along axis shuffled,
// numpy's permutation(x, axis).
func (g *Generator) PermutationOf(a *ndarray.Array, axis int) (*ndarray.Array, error) {
	if a.Ndim() == 0 {
		return nil, valueErr("axis %d is out of bounds for array of dimension 0", axis)
	}
	ax, err := normAxis(axis, a.Ndim())
	if err != nil {
		return nil, err
	}
	if a.Ndim() == 1 {
		c := a.Copy()
		return c, g.Shuffle(c, 0)
	}
	idx, _ := g.Permutation(a.Shape()[ax])
	d, _ := ndarray.Data[int64](idx)
	return takeAxis(a, d, []int{len(d)}, ax)
}

// Permuted returns a copy of a with its elements shuffled as one flat array,
// numpy's permuted(x) (axis=None).
func (g *Generator) Permuted(a *ndarray.Array) (*ndarray.Array, error) {
	c := a.Copy()
	s := reflect.ValueOf(storage(c))
	swap := reflect.Swapper(s.Interface())
	for i := s.Len() - 1; i >= 1; i-- {
		swap(i, int(interval(g.bg, uint64(i))))
	}
	return c, nil
}

// PermutedAxis returns a copy of a with each 1-d slice along axis shuffled
// independently, numpy's permuted(x, axis=axis); slices are taken in C order.
func (g *Generator) PermutedAxis(a *ndarray.Array, axis int) (*ndarray.Array, error) {
	ax, err := normAxis(axis, a.Ndim())
	if err != nil {
		return nil, err
	}
	c := a.Copy()
	s := reflect.ValueOf(storage(c))
	swap := reflect.Swapper(s.Interface())
	outer, n, inner := layout(c.Shape(), ax)
	for o := range outer {
		for in := range inner {
			base := o*n*inner + in
			for i := n - 1; i >= 0; i-- {
				j := int(interval(g.bg, uint64(i)))
				swap(base+i*inner, base+j*inner)
			}
		}
	}
	return c, nil
}

// ChoiceOptions are numpy's choice keywords. Their zero values are numpy's
// defaults: with replacement, uniform, axis 0, shuffled.
type ChoiceOptions struct {
	// NoReplace samples without replacement (numpy's replace=False).
	NoReplace bool
	// P are the probabilities of the entries; nil is uniform.
	P []float64
	// Axis is the axis of a to sample along.
	Axis int
	// NoShuffle leaves a sample without replacement in selection order
	// (numpy's shuffle=False).
	NoShuffle bool
}

// ChoiceN returns int64 indices sampled from arange(n), numpy's
// choice(n, size, replace, p, shuffle=...).
func (g *Generator) ChoiceN(n int64, o ChoiceOptions, size ...int) (*ndarray.Array, error) {
	if n <= 0 && ndarrayProd(size) != 0 {
		return nil, valueErr("a must be a positive integer unless no samples are taken")
	}
	idx, err := g.choiceIndices(n, o, size)
	if err != nil {
		return nil, err
	}
	return ndarray.FromSlice(idx, size...)
}

// Choice returns entries of a sampled along o.Axis, numpy's
// choice(a, size, replace, p, axis, shuffle): a's shape with that axis
// replaced by size.
func (g *Generator) Choice(a *ndarray.Array, o ChoiceOptions, size ...int) (*ndarray.Array, error) {
	if a.Ndim() == 0 {
		return nil, valueErr("a must be a sequence or an integer, not a 0-d array")
	}
	ax, err := normAxis(o.Axis, a.Ndim())
	if err != nil {
		return nil, err
	}
	n := int64(a.Shape()[ax])
	if n == 0 && ndarrayProd(size) != 0 {
		return nil, valueErr("a cannot be empty unless no samples are taken")
	}
	idx, err := g.choiceIndices(n, o, size)
	if err != nil {
		return nil, err
	}
	return takeAxis(a, idx, size, ax)
}

func ndarrayProd(size []int) int {
	p := 1
	for _, d := range size {
		p *= d
	}
	return p
}

// checkP validates choice's probabilities as numpy does.
func checkP(p []float64, n int64) error {
	if int64(len(p)) != n {
		return valueErr("a and p must have same size")
	}
	s := kahanSum(p)
	if math.IsNaN(s) {
		return valueErr("Probabilities contain NaN")
	}
	for _, v := range p {
		if v < 0 {
			return valueErr("Probabilities are not non-negative")
		}
	}
	if math.Abs(s-1.) > math.Sqrt(0x1p-52) {
		return valueErr("Probabilities do not sum to 1. See Notes section of docstring for more information.")
	}
	return nil
}

// cdfOf is numpy's cdf = p.cumsum(); cdf /= cdf[-1].
func cdfOf(p []float64) []float64 {
	cdf := make([]float64, len(p))
	acc := 0.0
	for i, v := range p {
		acc += v
		cdf[i] = acc
	}
	last := cdf[len(cdf)-1]
	for i := range cdf {
		cdf[i] /= last
	}
	return cdf
}

// searchRight is numpy's searchsorted(side='right').
func searchRight(cdf []float64, u float64) int64 {
	return int64(sort.Search(len(cdf), func(i int) bool { return cdf[i] > u }))
}

func (g *Generator) choiceIndices(n int64, o ChoiceOptions, size []int) ([]int64, error) {
	if o.P != nil {
		if err := checkP(o.P, n); err != nil {
			return nil, err
		}
	}
	count := ndarrayProd(size)
	if count < 0 {
		return nil, valueErr("negative dimensions are not allowed")
	}
	idx := make([]int64, count)
	if !o.NoReplace {
		if o.P != nil {
			cdf := cdfOf(o.P)
			for i := range idx {
				idx[i] = searchRight(cdf, g.bg.Float64())
			}
			return idx, nil
		}
		u := make([]uint64, count)
		fillUint64(g.bg, 0, uint64(n-1), false, u)
		for i := range u {
			idx[i] = int64(u[i])
		}
		return idx, nil
	}
	if int64(count) > n {
		return nil, valueErr("Cannot take a larger sample than population when replace is False")
	}
	if o.P != nil {
		return g.choiceWeighted(o.P, idx)
	}
	g.choiceFloyd(n, int64(count), !o.NoShuffle, idx)
	return idx, nil
}

// choiceWeighted is numpy's weighted sampling without replacement: draw,
// keep the first occurrence of each new index, zero the found weights,
// repeat.
func (g *Generator) choiceWeighted(p []float64, found []int64) ([]int64, error) {
	nz := 0
	for _, v := range p {
		if v > 0 {
			nz++
		}
	}
	if nz < len(found) {
		return nil, valueErr("Fewer non-zero entries in p than size")
	}
	p = append([]float64(nil), p...)
	nUniq := 0
	for nUniq < len(found) {
		x := make([]float64, len(found)-nUniq)
		for i := range x {
			x[i] = g.bg.Float64()
		}
		for _, f := range found[:nUniq] {
			p[f] = 0
		}
		cdf := cdfOf(p)
		seen := map[int64]bool{}
		for _, u := range x {
			k := searchRight(cdf, u)
			if !seen[k] {
				seen[k] = true
				found[nUniq] = k
				nUniq++
			}
		}
	}
	return found, nil
}

// choiceFloyd is numpy's unweighted sampling without replacement: a partial
// Fisher-Yates over arange(n) when the sample is a large part of a large
// population, otherwise Floyd's algorithm with a hash set, then (if shuffle)
// a Fisher-Yates over the sample.
func (g *Generator) choiceFloyd(n, size int64, shuffle bool, idx []int64) {
	cutoff := int64(20)
	if shuffle {
		cutoff = 50
	}
	if n > 10000 && size > n/cutoff {
		all := make([]int64, n)
		for i := range all {
			all[i] = int64(i)
		}
		shuffleInt(g.bg, all, max(n-size, 1))
		copy(idx, all[n-size:])
		return
	}
	setSize := uint64(1.2 * float64(size))
	mask := genMask(setSize)
	hash := make([]uint64, mask+1)
	for i := range hash {
		hash[i] = ^uint64(0)
	}
	for j := n - size; j < n; j++ {
		val := boundedUint64(g.bg, 0, uint64(j), false)
		loc := val & mask
		for hash[loc] != ^uint64(0) && hash[loc] != val {
			loc = (loc + 1) & mask
		}
		if hash[loc] == ^uint64(0) {
			hash[loc] = val
			idx[j-n+size] = int64(val)
		} else {
			loc = uint64(j) & mask
			for hash[loc] != ^uint64(0) {
				loc = (loc + 1) & mask
			}
			hash[loc] = uint64(j)
			idx[j-n+size] = j
		}
	}
	if shuffle {
		shuffleInt(g.bg, idx, 1)
	}
}

// shuffleInt is numpy's _shuffle_int: Fisher-Yates from the end down to
// first, with Lemire bounded draws.
func shuffleInt(bg BitGenerator, d []int64, first int64) {
	for i := int64(len(d)) - 1; i >= first; i-- {
		j := boundedUint64(bg, 0, uint64(i), false)
		d[i], d[j] = d[j], d[i]
	}
}
