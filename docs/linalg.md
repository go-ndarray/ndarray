# linalg — numpy.linalg in pure Go

Package [`linalg`](../linalg) gives ndarray arrays the functions of
`numpy.linalg`, written in Go without cgo or assembly of its own: the
matrix products it relies on are the ndarray package's packed SIMD GEMM.

| numpy.linalg | linalg |
|---|---|
| `solve(a, b)` | `Solve(a, b)` |
| `inv(a)` | `Inv(a)` |
| `det(a)`, `slogdet(a)` | `Det(a)`, `Slogdet(a)` |
| `cholesky(a, upper=…)` | `Cholesky(a, Lower \| Upper)` |
| `qr(a, mode)` | `QR(a, ModeReduced \| ModeComplete \| ModeR \| ModeRaw)` |
| `svd(a, full_matrices)`, `svdvals(a)` | `SVD(a, full)`, `SVDVals(a)` |
| `eigh(a, UPLO)`, `eigvalsh(a, UPLO)` | `Eigh(a, uplo)`, `Eigvalsh(a, uplo)` |
| `eig(a)`, `eigvals(a)` | `Eig(a)`, `Eigvals(a)` |
| `lstsq(a, b, rcond)` | `Lstsq(a, b)`, `LstsqRcond(a, b, rcond)` → `LstsqResult{X, Residuals, Rank, S}` |
| `pinv(a, rcond)` | `Pinv(a)`, `PinvRcond(a, rcond)` |
| `matrix_rank(A, tol)` | `MatrixRank(a)`, `MatrixRankTol(a, tol)` |
| `norm(x, ord, axis, keepdims)` | `Norm(x, ord, axes, keepdims)`; `ord` is `Ord{}` (None), `P(p)`, `Fro`, `Nuc` |
| `cond(x, p)` | `Cond(x, p)` |
| `matrix_power(a, n)` | `MatrixPower(a, n)` |
| `multi_dot(arrays)` | `MultiDot(arrays...)` |
| `LinAlgError` | `ErrLinAlg`, wrapped by `ErrSingular`, `ErrNotPosDef`, `ErrNoConvergence` |

Shape problems give `ErrShape` (numpy raises `LinAlgError` for some of them
and `ValueError` for others), bad options `ErrArgs`.

## Semantics

- **Stacks.** The last two axes are the matrix; leading axes are a stack,
  processed one matrix per goroutine. `Solve` broadcasts the stack
  dimensions of `a` and `b` as numpy's gufunc does; a 1-D `b` is one
  vector, any other `b` a stack of matrices (numpy 2's rule).
- **Dtypes.** As in numpy, everything is computed in double precision:
  float64 and complex128 directly, integers and bools promoted to float64,
  float32 and complex64 widened. Results have numpy's dtypes: single
  precision in, single precision out (rounded from the double result, which
  is what numpy does too — its linalg gufuncs also run in double); singular
  values, Hermitian eigenvalues, `logabsdet` and norms are real; `Eig` is
  always complex (numpy 2). `MatrixPower` with n ≥ 0 keeps an integer dtype.
- **Conventions.** LAPACK's: Householder reflectors with β = −sign(α)‖x‖, so
  R's diagonal can be negative as in numpy; ascending Hermitian eigenvalues;
  decreasing singular values; `Eig` vectors with unit norm and their largest
  component real, conjugate vectors for conjugate eigenvalues of a real
  matrix (dgeev's normalization).
- **Not unique.** Signs and phases of eigenvectors and singular vectors, and
  any orthonormal basis of a repeated eigenvalue's space, may differ from
  numpy's. Compare `A·v = λ·v`, `A = U·S·Vh`, not the vectors.

## Algorithms, and why

All are LAPACK's, re-expressed for row-major storage. Each is written once,
generic over `float64 | complex128`; Go 1.27 stencils the two and folds the
type switches of the helpers (`conj`, `abs`, …), so the real code pays
nothing for the complex case (measured: same speed as a hand-specialised
loop).

| operation | algorithm | where the time goes |
|---|---|---|
| `Solve`, `Inv`, `Det`, `Slogdet` | LU with partial pivoting, recursive (getrf2): halves the columns, so the trailing update and the triangular solves are GEMMs | GEMM |
| `Cholesky` | recursive potrf: `L21 = A21·L11⁻ᴴ`, `A22 −= L21·L21ᴴ` | GEMM |
| `QR` | blocked Householder (geqrf/orgqr, panels of 32, compact WY `I − V·T·Vᴴ`) | GEMM |
| `Eigh` | blocked tridiagonalization (latrd, panels of 32: half the flops in GEMM, half in matrix-vector products) + divide and conquer (stedc: Cuppen's split, deflation, secular equation, Gu–Eisenstat's recomputed vector) above order 25, implicit QL/QR (steqr) below | tridiagonalization |
| `Eigvalsh` | the same reduction + QL/QR without vectors | tridiagonalization |
| `SVD` | blocked Golub–Kahan bidiagonalization (labrd) + divide and conquer on the bidiagonal (bdsdc: Gu and Eisenstat's arrow matrix and secular equation) above order 25, implicit zero-shift/shifted QR (bdsqr, Demmel–Kahan) below; wide matrices factor Aᴴ | bidiagonalization |
| `SVDVals`, `MatrixRank`, `Cond` (2-norm), `Norm` (2, nuc) | the same reduction + bdsqr without vectors | bidiagonalization |
| `Eig`, `Eigvals` | balancing (gebal: permutations and powers of two), Hessenberg reduction (gehd2), Francis double-shift QR (dlahqr; single-shift zlahqr for complex input), eigenvectors by back-substitution on the Schur form with overflow scaling (trevc), back-transformed by one GEMM, gebak, dgeev's normalization | QR iteration |
| `Lstsq` | SVD, `x = V·S⁺·Uᴴ·b` over singular values above `rcond·σ₁`; the residuals from the Householder QR of `a` (the tail of `Qᴴb`, free of the cancellation in `b − a·x`) | SVD |
| `Pinv` | SVD, `V·S⁺·Uᴴ` | SVD |
| `Norm`, `MatrixPower`, `MultiDot` | direct; 2-norms rescaled against overflow; repeated squaring in numpy's order; numpy's matrix-chain dynamic programme | — |

**The SVD: why Golub–Kahan with QR, and divide and conquer.** The choice
was between Golub–Kahan bidiagonalization followed by the implicit-shift QR
iteration, and one-sided Jacobi. One-sided Jacobi is the more accurate in a
strong sense — it computes the small singular values of a matrix whose
columns are badly scaled to high relative accuracy (Demmel–Veselić) — but it
needs 5–10 sweeps of O(n³) rotations, several times the work, and it is not
what numpy computes. Golub–Kahan reduction is backward stable: the computed
singular values are those of `A + E` with `‖E‖ ≈ ε‖A‖`, so each is within
about `ε·σ₁` of the truth — numpy's (gesdd's) accuracy, exactly. The QR
iteration (bdsqr, with Demmel and Kahan's zero shift) then finds the
singular values of the bidiagonal to high relative accuracy, so it adds
nothing to that error. It is used at the leaves of the divide and conquer
and whenever the vectors are not wanted. With vectors, the QR iteration's
plane rotations cost O(n³) of non-vectorized work (5.8 s at n = 1024 in the
first version); divide and conquer, LAPACK's gesdd algorithm and the one
numpy calls, puts that work into GEMMs (0.26 s now). Its singular vectors
stay orthogonal because the updating vector is recomputed from the computed
roots (Gu and Eisenstat), and the secular equation is solved with each
distance `dᵢ − σ` formed from the nearest pole, in the squared form
`(dᵢ − σ)(dᵢ + σ)`.

**Hermitian eigenproblems** follow the same pattern: the tridiagonal QL/QR
iteration (steqr) is used without vectors and for orders up to 25, divide
and conquer (stedc) above, which took n = 1024 from 3.6 s to 0.13 s.

**General eigenproblems** use the classical double-shift QR (dlahqr), not
LAPACK's multishift QR with aggressive early deflation (dlaqr0–5). That is
the largest gap to numpy, at large orders (below).

## Accuracy and verification

**Against numpy.** `linalg/testdata/cases.py` generates 1,040 cases with
numpy 2.5.3 (`uv run --with numpy python linalg/testdata/cases.py >
linalg/testdata/cases.json`; the JSON is embedded in the test binary, so the
emulated CI architectures read it from anywhere). They cover every
function; float64, float32, complex128, complex64, int32/int64 and bool
inputs; 0×0, 1×1, tall, wide, stacked and broadcast shapes; Hilbert
matrices (κ up to 10¹⁶), graded and rank-deficient matrices, exactly
singular ones, non-symmetric matrices with complex eigenvalues, Jordan
blocks and other defective matrices, NaN and Inf inputs; and every error
numpy raises on them (124 cases). The test checks, for each case:

- errors: an error of the same kind (`ErrSingular`, `ErrNotPosDef`,
  `ErrNoConvergence`, `ErrLinAlg` for non-finite input to `Eig`,
  `ErrShape`, `ErrArgs`);
- dtype and shape of every result, exactly;
- unique values (solutions, inverses, determinants, Cholesky factors,
  singular values, eigenvalues, least-squares solutions and residuals,
  pseudo-inverses, ranks, norms, condition numbers, powers, products)
  elementwise, within `32·n·ε·κ` of the largest entry of the result, with
  `κ` the condition number of the problem (1 for singular values, Hermitian
  eigenvalues and norms; `cond(A)` for solves and inverses; the effective
  condition over the kept singular values for `Lstsq` and `Pinv`, squared for
  the residuals), plus one float32 rounding for single-precision results;
- factors that are not unique through their identities: `‖QᴴQ − I‖`,
  `‖A − QR‖/‖A‖`, R upper triangular; `‖UᴴU − I‖`, `‖VhVhᴴ − I‖`,
  `‖A − U·S·Vh‖/‖A‖`; `‖VᴴV − I‖`, `‖AV − VΛ‖/‖A‖` with ascending
  eigenvalues; for `Eig`, `‖Av − λv‖/‖A‖` per vector, unit norm, largest
  component real, conjugate pairs exactly conjugate — all within
  `32·n·ε` (`ε` of the result's dtype);
- general eigenvalues matched to numpy's nearest unused one, within 100
  times how far numpy's own eigenvalues move when each entry is perturbed by
  10⁻¹⁴ of itself (an empirical condition number, recorded in the fixture),
  so a Jordan block's eigenvalue, accurate only to `ε^(1/4)`, is judged by
  what is achievable.

914 cases are value-checked and 124 are errors; the 2 deliberate departures
below are logged.

**Beyond the fixture** (`properties_test.go`, `kernels_test.go`): the same
identities at orders 100 and 131 (above the blocking and divide-and-conquer
thresholds), real and complex, tall and wide; a stack of 16 matrices run in
parallel equals the matrices run one by one; repeated eigenvalues and
singular values, rank-deficient and zero matrices (the deflation paths);
cyclic permutations, Grcar and Frank matrices, orthogonal matrices,
high-order Jordan blocks and coupled rotation blocks (exceptional shifts,
overflow rescaling in the back-substitution); balancing at the limits of the
exponent range; every iteration forced to fail (`ErrNoConvergence` from
each function); and the 2×2 kernels (laev2, las2, lasv2, lanv2, lartg,
solve2) checked through their defining identities on the inputs that reach
their special cases.

**Negative controls.** Dropping the row interchange in the LU leaf (no
partial pivoting) fails 64 fixture cases — 23 `solve`, 17 `inv`, 10 `det`,
10 `slogdet`, 4 `matrix_power` — and the order-100 identity test. Loosening
the secular-equation solver's stopping test by 10⁸ fails the four order-34
`eigh` cases (eigenvalues off by ~10⁻¹¹ relative) and the zero-diagonal
bidiagonal divide-and-conquer test (`‖B − USVᵀ‖` of 10⁻⁸).

**Coverage** is 100% of statements under the CI gate (`go test -race
-coverpkg=./... ./...`).

## Speed against numpy

Median wall time of `BenchmarkLinalg` against numpy 2.5.3 (`la.solve(a,
ones)`, `la.inv`, `la.cholesky` of `A·Aᵀ + nI`, `la.qr` reduced,
`la.svd(full_matrices=False)`, `la.eigh` of `A + Aᵀ`, `la.eig`), n×n
standard normal matrices, Apple M-series (16 cores, arm64), three
interleaved rounds. numpy's wheel uses **Apple Accelerate**. The host was
shared and heavily loaded (load average 16–23 on 16 cores during the runs),
so absolute numbers are noisy; the ratios held across rounds.

| op | n | Go (ms) | numpy (ms) | Go / numpy |
|---|---|---|---|---|
| solve | 64 | 0.046 | 0.022 | 2.1 |
| solve | 256 | 0.99 | 0.25 | 3.9 |
| solve | 1024 | 21.2 | 10.9 | 1.9 |
| inv | 64 | 0.11 | 0.029 | 3.7 |
| inv | 256 | 2.19 | 0.58 | 3.8 |
| inv | 1024 | 39.4 | 23.7 | 1.7 |
| cholesky | 64 | 0.030 | 0.009 | 3.3 |
| cholesky | 256 | 0.54 | 0.24 | 2.2 |
| cholesky | 1024 | 10.9 | 6.6 | 1.7 |
| qr | 64 | 0.15 | 0.083 | 1.8 |
| qr | 256 | 2.32 | 1.57 | 1.5 |
| qr | 1024 | 54.7 | 37.3 | 1.5 |
| svd | 64 | 0.78 | 0.30 | 2.6 |
| svd | 256 | 13.3 | 5.25 | 2.5 |
| svd | 1024 | 263 | 123 | 2.1 |
| eigh | 64 | 0.41 | 0.22 | 1.9 |
| eigh | 256 | 6.54 | 3.12 | 2.1 |
| eigh | 1024 | 130 | 78 | 1.7 |
| eig | 64 | 1.07 | 1.06 | 1.0 |
| eig | 256 | 55 | 19.4 | 2.8 |
| eig | 1024 | 5060 | 712 | 7.1 |

**numpy is faster on single matrices**, by 1.5–4× (7× for `eig` at 1024).
Most of that is the matrix product underneath: on the same host the ndarray
GEMM takes 18 ms for a 1024² product where Accelerate (which uses the
matrix coprocessor) takes 3.8 ms, 0.37 ms against 0.10 ms at 256. The
blocked algorithms here spend most of their time in that GEMM, so their gap
to numpy is smaller than the GEMM's. The exceptions:

- `eig` at large n: the classical double-shift QR is O(10n³) of scalar
  work; LAPACK's multishift QR with aggressive early deflation moves it into
  GEMMs. This is the main follow-up.
- small orders (64): fixed costs (conversion of the input, allocation,
  the packing of a small GEMM) weigh more than the arithmetic.

**Stacks of small matrices run in parallel** — numpy's gufuncs loop over
them on one core — and there this package is faster (median of 3, same
host):

| op | stack | Go (ms) | numpy (ms) | numpy / Go |
|---|---|---|---|---|
| inv | 10000 × 4×4 | 0.94 | 3.83 | 4.1 |
| det | 10000 × 4×4 | 0.81 | 2.12 | 2.6 |
| eigh | 10000 × 4×4 | 3.0 | 13.8 | 4.6 |
| svd | 10000 × 4×4 | 5.3 | 29.5 | 5.6 |
| eig | 10000 × 4×4 | 4.9 | 55.7 | 11 |
| inv | 1000 × 32×32 | 3.4 | 8.5 | 2.5 |
| det | 1000 × 32×32 | 1.6 | 4.0 | 2.5 |
| eigh | 1000 × 32×32 | 21.3 | 50.4 | 2.4 |
| svd | 1000 × 32×32 | 44.3 | 75.9 | 1.7 |
| eig | 1000 × 32×32 | 29.7 | 232 | 7.8 |

Reproduce: `go test -run XXX -bench 'Linalg|Stacked' ./linalg`, and the
numpy side with `uv run --with numpy==2.5.3 python docs/bench_linalg_numpy.py`
(the same sizes and distributions; the values differ, being from another
generator).

## Differences from numpy

| numpy | here | why |
|---|---|---|
| `det` is `sign·exp(logabsdet)` | the product of the pivots, accumulated as mantissa and exponent | exact to rounding, no spurious overflow or underflow of intermediate products, and no `|logdet|·ε` error from `exp` |
| `norm` of a vector with entries near 10²⁰⁰ overflows to `inf` | the 2-norms are rescaled (`1.414e200` for `[1e200, 1e200]`) | the true value is representable |
| float32 norms accumulate in float32 | in float64, then rounded | more accurate; within a float32 rounding of numpy |
| `cond(singular, 'nuc')` raises `LinAlgError` (the SVD of its NaN "inverse" fails) | `+Inf`, as for every other order | a singular matrix has infinite condition number |
| `eigh` of a matrix with a NaN returns NaN eigenvalues | `ErrNoConvergence` | the iteration cannot proceed; numpy's `svd` and `eig` refuse NaN too |
| `LinAlgError` for a wrong rank or a non-square matrix | `ErrShape` | a shape error, not a numerical failure |
| `svd(…, hermitian=True)`, `pinv(…, hermitian=True)`, `matrix_rank(…, hermitian=True)`, `matrix_rank(rtol=)`, `pinv(rtol=)` | not provided (`PinvRcond(a, max(M,N)·ε)` is `rtol=None`) | follow-ups |
| `tensorsolve`, `tensorinv`, `cross`, `outer`, `trace`, `diagonal`, `matmul`, `vecdot`, `matrix_norm`, `vector_norm`, `matrix_transpose` | not in this package (`ndarray` has `MatMul`, `Outer`, `Transpose`; `Norm` covers both norms) | follow-ups |
| `qr` modes `'full'` and `'economic'` (deprecated) | not provided | deprecated in numpy |

## What is left

- Multishift QR with aggressive early deflation for `Eig` (the 7× gap at
  n = 1024), and a blocked Hessenberg reduction.
- A faster small-matrix path (n ≤ 64): unblocked kernels without the GEMM
  packing, fewer copies.
- The `hermitian` options and `rtol` arguments above, and the tensor
  functions.
- `Eigh` on NaN input returning NaNs like numpy, if that is preferred.
