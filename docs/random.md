# random — numpy.random, the same numbers

Package [`random`](../random) is `numpy.random` over go-ndarray arrays. Its goal
is migration: a notebook moved from NumPy to Go keeps its numbers for the same
seed.

```go
rng := random.DefaultRNG(42)          // np.random.default_rng(42)
x, _ := rng.StandardNormal(3, 4)      // rng.standard_normal((3, 4))
k, _ := rng.Integers(0, 10, 5)        // rng.integers(0, 10, 5)

random.Seed(0)                        // np.random.seed(0)
u, _ := random.Rand(3)                // np.random.rand(3)
```

`Example`, `ExampleSeed` and `ExampleGenerator_ChoiceN` in
[`example_test.go`](../random/example_test.go) print NumPy's own numbers.

## What is implemented

| NumPy | here |
|---|---|
| `SeedSequence(entropy, spawn_key=, pool_size=)`, `.pool`, `.generate_state(n, uint32/uint64)`, `.spawn(n)` | `NewSeedSequence(v...)`, `NewSeedSequenceBig`, `NewSeedSequenceWith`, `Pool`, `GenerateState`, `GenerateState64`, `Spawn` |
| `PCG64`, `PCG64DXSM`, `MT19937`, `Philox` (also `counter=`, `key=`), `SFC64` | `NewPCG64` … `NewSFC64`, `New…FromSeedSequence`, `NewPhiloxKey` |
| `.jumped(n)`, `.advance(delta)`, `.random_raw(n)`, `.spawn(n)`, `.state` | `Jumped`, `Advance(*big.Int)`, `RandomRaw`, `Spawn`, `State`/`SetState` (typed structs) |
| `default_rng(seed)`, `Generator(bitgen)`, `.spawn(n)` | `DefaultRNG(seed...)`, `NewGenerator`, `Spawn` |
| `random(size, dtype)` | `Random(size...)`, `RandomOf(dt, size...)` |
| `integers(low, high, size, dtype, endpoint)` | `Integers(low, high, size...)`, `IntegersOf(dt, low, high, endpoint, size...)`, `IntegersUint64` |
| `standard_normal`, `normal`, `standard_exponential(method=)`, `exponential`, `uniform`, `standard_gamma(dtype=)`, `gamma` | `StandardNormal[Of]`, `Normal`, `StandardExponential[Of]`, `Exponential`, `Uniform`, `StandardGamma[Of]`, `Gamma` |
| `beta`, `chisquare`, `f`, `noncentral_chisquare`, `noncentral_f`, `standard_t`, `standard_cauchy`, `lognormal`, `laplace`, `logistic`, `gumbel`, `weibull`, `pareto`, `power`, `rayleigh`, `wald`, `triangular`, `vonmises` | same names in Go case |
| `binomial`, `negative_binomial`, `poisson`, `geometric`, `zipf`, `logseries` | same names, int64 results |
| `choice(a, size, replace, p, axis, shuffle)` | `ChoiceN(n, opts, size...)`, `Choice(a, opts, size...)` |
| `shuffle(x, axis)`, `permutation(n)`, `permutation(x, axis)`, `permuted(x, axis=)` | `Shuffle`, `Permutation`, `PermutationOf`, `Permuted`, `PermutedAxis` |
| `bytes(n)`, `dirichlet(alpha, size)`, `multinomial(n, pvals, size)` | `Bytes`, `Dirichlet`, `Multinomial` |
| `RandomState(seed)`, `.seed`, `.get_state`/`.set_state` | `NewRandomState(uint32)`, `NewRandomStateArray`, `NewRandomStateFrom(bitgen)`, `Seed`, `SeedArray`, `GetState`/`SetState` |
| `rand`, `randn`, `randint`, `random_sample`, `normal`, `uniform`, `choice`, `shuffle`, `permutation`, `binomial`, `poisson`, `standard_normal`, `standard_exponential`, `exponential`, `standard_gamma`, `gamma`, `chisquare`, `lognormal`, `multinomial`, `bytes` | `RandomState` methods of the same names |
| `np.random.seed`, `rand`, `randn`, `randint`, `random_sample`, `normal`, `uniform`, `choice`, `shuffle`, `permutation`, `binomial`, `poisson` | package functions over a shared `RandomState` behind a mutex |

The size comes last; no size gives a 0-d array holding what NumPy returns as a
scalar (`size=None`), which draws exactly the same values. `ChoiceOptions`'
zero value is NumPy's defaults (`NoReplace`, `NoShuffle` are the negations).

**Not implemented:** array-valued (broadcast) distribution parameters and
integer bounds, `out=`, `hypergeometric`, `multivariate_normal`,
`multivariate_hypergeometric`, the `RandomState` samplers not listed above
(`beta`, `f`, `pareto` …: NumPy 1.16's versions differ from `Generator`'s and
were not ported), `tomaxint`, `random_integers`, pickling, and `float16`.

## Verification

`random/testdata/cases.py` runs NumPy 2.5.3 and records:

- **SeedSequence**: the pool, 9 uint32 and 5 uint64 state words, and the pools
  and spawn keys of 4 children, for 10 entropies (0, 42, 2⁶⁴−1, `[1,2,3]`,
  2¹²⁸+12345, a 10-word list, `[]`, pool size 8, spawn keys with a 2⁴⁰ entry,
  a key after a 6-word entropy);
- **bit generators**: for each of the five and six seeds (0, 1, 42, 2⁶⁴−1,
  `[1,2,3]`, 2¹²⁸+12345), the seeded state, 100 raw outputs (700 for MT19937,
  which crosses a regeneration), `jumped(1)`, `jumped(3)`, `advance(2¹⁰⁰+7)`,
  `advance(-3)`, two spawned children; Philox from an explicit counter and key
  and across a counter carry; legacy MT19937 seeding from 6 integer and array
  seeds, and its jump;
- **2,222 scripts, 2,492 calls, 81,806 values**: every Generator function on
  PCG64 with every seed, every dtype and bound case of `integers` (9 dtypes,
  open and closed, full ranges, ranges of 2³²−1, 2³², 3·2³⁰, 3·2⁶², empty
  ranges); a subset on the four other bit generators; an interleaved script
  per generator and seed (the buffered upper half of a 64-bit draw outlives a
  call); `RandomState` on 6 seeds (including a 700-word array);
- **50 long runs as SHA-256 digests** (4.49 million values): `random`,
  `standard_normal` and `standard_exponential` (both dtypes, 200,000 each,
  which reaches the ziggurat tails and wedges), `integers` (Lemire with
  rejections, 8- and 16-bit buffering, bool), `binomial` (inversion and
  BTPE), `poisson` (both methods), `choice` with and without replacement, and
  `RandomState`'s `rand`, `randint`, `binomial`, `poisson`, `uniform`.

The fixture is embedded, so CI's emulated targets check it too. It passes,
unchanged, on darwin/arm64, darwin/amd64 (Rosetta), and linux 386, arm,
riscv64, ppc64le and **s390x** (big-endian: `Bytes` gives NumPy's
little-endian bytes) under qemu.

**Integers, raw output, and every sampler that is plain arithmetic match NumPy
to the bit** — including the ziggurat normal and exponential (tails included,
over 200,000 draws), binomial, Poisson, geometric, choice, shuffle and the
legacy `randint`/`rand`/`binomial`/`poisson`.

### Negative controls

Each mutation below was made on purpose and the suite run:

| mutation | result |
|---|---|
| PCG64 multiplier, low word + 2 | 1,155 failures |
| SeedSequence `MIX_MULT_L` + 2 | 1,998 failures |
| SeedSequence `MULT_A` + 2 | 2,005 failures |
| one ziggurat `ki_double` entry losing its last hex digit (an actual bug of the first table converter, which stripped a trailing `F` as if it were a C suffix) | 15 failures, all in the long runs |
| `RandomState.randint` with Lemire instead of masked rejection | 44 failures |
| MT19937 seeded with `pos = 624` instead of NumPy's `623` | 266 failures |
| legacy BTPE adding the last two Stirling terms with the corrected signs | the `RandomState` binomial digests fail |
| legacy BTPE constant 13680 → 13860 | **not detected**: the term changes by ~2·10⁻⁹ and flips no decision in 10⁶ draws at any of four (n, p) tried |

## Where the numbers can differ: the C library

Some samplers call `log`, `exp`, `log1p`, `expm1`, `pow`, `cos` or `acos`.
NumPy takes them from the C library; Go's `math` package does not round them
the same way. Measured against the libm of the x86-64 NumPy used for the
fixture (Apple's), Go's `math` differs in 6.5% (`log`) to 71% (`pow`) of
random arguments. Used as-is, that made 3% of `np.random.randn` values and
10% of `lognormal` values one ULP off.

So `random` rounds `log`, `log1p`, `exp`, `expm1` and `pow` **correctly**
([`libm.go`](../random/libm.go)): a fast evaluation to about 2⁻⁶⁸ with Ziv's
rounding test, and a double-double fallback (about 2⁻¹⁰⁰) for the one argument
in ~2¹⁵ that needs it. Checked against mpmath at 300 bits on 63,000 arguments
(0 misrounded; the committed `libm.json` keeps 8,400), and the fast paths
against the fallback on 400,000 more. The code converts every product
explicitly, so Go cannot fuse it into an FMA, and the result is the same on
every architecture.

The C libraries round these functions correctly almost always — Apple's x86-64
libm misrounds 0.06% (`log`) to 0.3% (`exp`) of arguments; `pow(x, 0.5)` is
misrounded for 0.19% (it is not `sqrt`). What remains, on 10⁶ draws against
x86-64 NumPy, seed 0:

| sampler | values that differ (of 10⁶) | max ULP | with Go's `math` instead |
|---|---:|---:|---:|
| `RandomState.randn` | 294 | 3 | 29,631 |
| `RandomState.normal(1, 3)` | 217 | 32¹ | 22,429 |
| `RandomState.standard_gamma(3)` | 72 | 6 | 5,762 |
| `RandomState.standard_exponential` | 664 | 1 | 68,249 |
| `standard_gamma(0.5)` (pow) | 1,870 | 3 | 16,212 |
| `beta(0.5, 0.5)` (pow) | 1,693 | 4 | 1,693 |
| `lognormal(0.5, 0.25)` | 2,263 | 1 | 99,340 |
| `laplace` / `logistic` | 690 / 601 | 1 | 68,357 / 68,518 |
| `gumbel` | 1,297 | 2049¹ | 131,066 |
| `standard_exponential(method="inv")` | 692 | 1 | 68,383 |
| `pareto(3)` (expm1) | 4,645 | 1 | 65,593 |
| `weibull(2)` (pow) | 1,777 | 1 | 1,779 |
| `vonmises(0.5, 2)` (cos, acos) | 126,540 | 2¹⁸ ¹ | 126,540 |
| `standard_normal`, `standard_gamma(3)`, `beta(2, 3)`, `chisquare`, `standard_t`, `binomial`, `poisson(3)`, `poisson(50)`, `geometric(0.05)` | **0** | — | 0 – 2 |

¹ Large ULP counts are absolute errors of about one ULP of the inner value,
measured relative to a result near zero (`1 + 3z` for `z ≈ −1/3`;
`log(-log(U))` for `U` near 1/e; an angle wrapped to near 0).

Each of the residual differences checked by hand (with mpmath) is a value
NumPy's libm rounded incorrectly. `vonmises` keeps Go's `cos` and `acos`:
Apple's `cos` itself misrounds 17% of arguments in [0, π], so rounding
correctly would not bring it closer. No difference has ever changed an
accept/reject decision — every integer-valued sampler matches exactly — and
the test bounds each float sampler's ULP distance individually
(`libmULP` in `numpy_test.go`).

## NumPy does not agree with itself

The same fixture generated by NumPy 2.5.3's **aarch64** macOS build differs from
its x86-64 build: the C compiler fuses `a*b + c` into one FMA there (clang's
default `-ffp-contract=on`), and Apple's aarch64 libm rounds differently.

| | values that differ, aarch64 vs x86-64 NumPy |
|---|---|
| `Generator.uniform` | 20.8% of 200,000 (`low + range*U` is fused) |
| `RandomState.randn` | 865 of 6,072 (14%) |
| `RandomState.normal` | 1,063 of 4,048 (26%) |
| `RandomState.standard_gamma` | 481 of 4,108 |
| `Generator.standard_gamma`, `chisquare`, `f`, `noncentral_*`, `beta`, `gamma`, `lognormal`, `standard_t`, `vonmises` | a few % |
| digests of `Generator.normal`, `uniform`, `standard_gamma` | differ |

This package follows the C source operation by operation without fusing,
which is what NumPy's x86-64 builds (Linux, Windows, Intel Macs) compute, and
it gives the same numbers on every architecture. A notebook run on an Apple
silicon Mac or an aarch64 Linux box will see NumPy's last bits move in those
samplers; that is NumPy, not a migration error.

## Deliberate differences

| NumPy | here | why |
|---|---|---|
| `randint`'s default dtype is C `long`: int64 on Linux/macOS, **int32 on Windows** | always int64 | the same numbers on every platform; `RandintOf(ndarray.Int32, …)` for Windows' |
| `size=None` returns a Python scalar | a 0-d array | one return type; same draws |
| `shuffle` works on views | `Shuffle` needs an array that owns its storage (`ErrNotContiguous` for a view) | ndarray hands out a view's elements as a copy |
| `permuted(x)` of a Fortran-ordered `x` shuffles in memory order | always C order | ndarray arrays are row-major |
| NumPy's `(int)` of ±∞ in rejection branches is undefined C | the same branch is rejected (the `v == 0` tests) | — |
| `zipf`'s `X > LONG_MAX` rejection | kept as a condition, unreachable short of `pow` rounding past 2⁶³ | — |
| `exp` of an argument whose result is subnormal | rounded twice (to 53 bits, then to the subnormal) | only below 2⁻¹⁰²² |

## Performance

Apple M4 Max, 16 cores, shared and loaded (load average 10.7–14.4 during the
runs). Seven interleaved rounds: the Go benchmark (`go test -bench`, 20
iterations) then NumPy 2.5.3 aarch64 (median of 20 calls); medians of the
rounds, in milliseconds for 10⁶ values:

| | Go | NumPy | Go / NumPy | Go range | NumPy range |
|---|---:|---:|---:|---|---|
| `random(10**6)` | 3.51 | 3.13 | 1.12 | 3.47–3.83 | 3.09–3.41 |
| `random(10**6, dtype=float32)` | 1.89 | 1.64 | 1.16 | 1.88–2.24 | 1.62–1.65 |
| `integers(0, 1000, 10**6)` | 2.31 | 1.70 | 1.36 | 2.29–2.32 | 1.68–1.71 |
| `standard_normal(10**6)` | 4.12 | 3.68 | 1.12 | 4.07–4.16 | 3.67–3.72 |
| `RandomState(0).randn(10**6)` | 20.36 | 9.65 | 2.11 | 19.99–20.56 | 9.61–9.68 |

Go is **slower everywhere**: 12–36% for the Generator, 2.1× for the legacy
`randn`. The Generator gap is the 128-bit LCG step (3.3 ns per output, a
dependent multiply chain) plus zeroing the result (`ZerosOf`; NumPy's
`np.empty` does not); the default PCG64 already gets direct, inlinable calls
for these four functions. `randn` pays for a correctly rounded `log` on every
pair (6 ns against 3.4 ns for `math.Log`) and for MT19937 behind an
interface. The samplers that call `pow`/`exp` per draw cost about what
`math.Pow`/`math.Exp` would (19.5 vs 18.4 ns; 6.3 vs 3.9 ns).

## Regenerating

```sh
uv run --python cpython-3.12-macos-x86_64 --with numpy==2.5.3 python random/testdata/cases.py > random/testdata/cases.json
uv run --with mpmath python random/testdata/libm.py > random/testdata/libm.json
uv run --with mpmath python random/testdata/libm_tables.py > random/zz_libm_tables.go
python3 random/testdata/zig2go.py <numpy>/numpy/random/src/distributions/ziggurat_constants.h random/zz_ziggurat.go
```

`random/testdata/quant.py` and `TestQuantifyLibm` (run with
`RANDOM_QUANT_DIR`) reproduce the 10⁶-draw table.
