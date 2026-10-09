// Package random is numpy.random for go-ndarray: the same numbers as NumPy
// for the same seed.
//
// A notebook that does
//
//	rng = np.random.default_rng(42)
//	x = rng.standard_normal((3, 4))
//
// becomes
//
//	rng := random.DefaultRNG(42)
//	x, _ := rng.StandardNormal(3, 4)
//
// and x holds NumPy's twelve values, bit for bit. Old code that calls
// np.random.seed(0) and np.random.rand(...) has random.Seed(0) and
// random.Rand(...), over the legacy RandomState, with the same numbers too.
//
// # What is here
//
//   - SeedSequence, NumPy's entropy mixer: pool, GenerateState, Spawn.
//   - The bit generators PCG64 (the default), PCG64DXSM, MT19937, Philox and
//     SFC64, with Jumped and Advance where NumPy has them, and RandomRaw.
//   - Generator (DefaultRNG, NewGenerator): Random, Integers, Uniform,
//     StandardNormal/Normal (ziggurat), StandardExponential/Exponential,
//     StandardGamma/Gamma, Beta, ChiSquare, F, Binomial, Poisson, Geometric,
//     Lognormal, Laplace, Logistic, Gumbel, Weibull, Pareto, Power, Rayleigh,
//     Wald, Triangular, VonMises, StandardT, StandardCauchy, Zipf, Logseries,
//     NegativeBinomial, the noncentral chi-square and F, Choice, Shuffle,
//     Permutation, Permuted, Bytes, Dirichlet and Multinomial.
//   - RandomState, the legacy generator (MT19937, the polar Gaussian, NumPy
//     1.16's samplers), and the module functions Seed, Rand, Randn, Randint,
//     RandomSample, Normal, Uniform, Choice, Shuffle, Permutation, Binomial
//     and Poisson over a shared one.
//
// Results are arrays with NumPy's dtypes and shapes: the size arguments come
// last (numpy's size), and no size gives a 0-d array holding what NumPy
// returns as a scalar. Parameters are scalars (NumPy also broadcasts array
// parameters; that is not implemented here).
//
// # Exactness
//
// Integers, raw bit-generator output and every sampler that uses only
// arithmetic match NumPy to the bit. Samplers that call log, exp, pow ...
// match it except where NumPy's C library rounds those functions
// incorrectly: this package rounds them correctly, the C libraries almost
// always do, and the remainder is a fraction of a percent of values, one
// ULP apart. docs/random.md has the measurements, including the ones where
// NumPy's own aarch64 builds disagree with its x86-64 builds.
//
// Generators are not safe for concurrent use; give each goroutine its own
// (Generator.Spawn).
package random
