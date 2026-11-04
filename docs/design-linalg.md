# Design: the dense linear-algebra foundation (`scipy.linalg` side)

Status: design, no code yet. Scope: what [coverage.md](coverage.md) slices 6–8
list as missing and the go-scikit roadmap (`COVERAGE.md`, "1. Shared
foundations"; `PLAN.md` item 10) puts first:

- **(a) exports of existing internals under `scipy.linalg` names**: `lu`,
  `lu_factor`/`lu_solve`, `cho_factor`/`cho_solve` (and the banded pair),
  `solve_triangular`, `qr(pivoting=True)`, `hessenberg`, `schur`/`rsf2csf`,
  `matrix_balance`;
- **(b) matrix functions**: `expm`, `logm`, `sqrtm`, `fractional_matrix_power`,
  `funm`, `signm`, `expm_frechet`, `expm_cond`, `cosm`/`sinm`/`tanm`,
  `coshm`/`sinhm`/`tanhm`;
- **(c) tridiagonal and banded**: `eigh_tridiagonal`, `eigvalsh_tridiagonal`,
  `eig_banded`, `eigvals_banded`, `solve_banded`, `solveh_banded`;
- **(d) generalized Schur**: `qz`, `ordqz`;
- **MATLAB delta**: `expm`, `logm`, `sqrtm`, `funm`, `schur`, `ordschur`, `qz`,
  `ordqz`, `balance`.

Consumers: go-statistics/models (pivoted QR, triangular solve), go-sparse
(`expm`, `onenormest`), go-signal (`lsim` needs `expm`), orthogonal
polynomials and Gauss rules in special/poly and integrate (tridiagonal
eigenvalues), Riccati/Lyapunov/Sylvester solvers later (`schur`, `qz`).

## 1. Sources

Opened for this design (S-numbers are cited below):

| | source | what was read |
|---|---|---|
| S1 | SciPy v1.18.1 `scipy/linalg/_matfuncs.py` | `expm`, `sqrtm`, `logm`, `cosm`…`tanhm`, `funm`, `signm`, `_maybe_real` |
| S2 | SciPy v1.18.1 `scipy/linalg/src/_matfuncs_expm.c` | `dnorm1`, `dnorm1est` (dlacn2), `pick_pade_structure_d`, `pade_UV_calc_{s,d,c,z}`, `matrix_exponential_d` |
| S3 | SciPy v1.18.1 `scipy/linalg/src/_matfuncs_sqrtm.c` | `matrix_squareroot_d`, `sqrtm_recursion_d` |
| S4 | SciPy v1.18.1 `_matfuncs_sqrtm.py`, `_matfuncs_sqrtm_triu.py` | `_sqrtm_triu` (blocked, used by `logm`), `within_block_loop` |
| S5 | SciPy v1.18.1 `_matfuncs_inv_ssq.py` | `_logm`, `_logm_triu`, `_inverse_squaring_helper`, `_fractional_matrix_power` (call graph) |
| S6 | SciPy v1.18.1 `_expm_frechet.py` | `expm_frechet` (SPS / blockEnlarge), `expm_cond` via the Kronecker form |
| S7 | SciPy v1.18.1 `_decomp_schur.py`, `_decomp_qz.py` | `schur` (gees, sort), `rsf2csf`, `qz` (gges, `sort_t=0`), `ordqz` (gges then tgsen `ijob=0`) |
| S8 | SciPy v1.18.1 `_decomp.py` | `eig_banded` (sbevd/hbevd, sbevx/hbevx with `abstol=2·lamch('s')`), `eigh_tridiagonal` drivers, `hessenberg` (gebal, gehrd, orghr) |
| S9 | SciPy v1.18.1 `_basic.py`, `_decomp_lu.py`, `_decomp_cholesky.py`, `_decomp_qr.py`, `src/_linalg_lu_det.hh`, `src/_linalg_qr.hh`, `src/_linalg_cholesky.hh` | driver of each export (table in §3) |
| S10 | SciPy v1.18.1 `scipy/special/_orthogonal.py` | `_gen_roots_and_weights` → `linalg.eigvals_banded` (line 180), used by `roots_legendre` (2628) and so by `logm` |
| S11 | SciPy v1.18.1 `scipy/sparse/linalg/_onenormest.py` | `t >= n` exact branch (93), `np.random.randint` columns (214) |
| S12 | Reference-LAPACK v3.12.1 `SRC/`: `dgees.f`, `dhseqr.f`, `iparmq.f`, `ilaenv.f`, `dtrsen.f`, `dtrsyl.f`, `dgges.f`, `dtgsen.f`, `dstevd.f`, `dsbevd.f`, `dstebz.f`, `dstein.f`, `dstemr.f`, `dgeqp3.f`, `dlapy2.f` | call graphs and tuning constants quoted in §3 |
| S13 | Al-Mohy, Higham, *A New Scaling and Squaring Algorithm for the Matrix Exponential*, SIMAX 31(3), 2009 (MIMS EPrint 2009.9) | §5 pp. 982–983: the choice of m and s, `ell`, θ₁₃ = 4.25, Algorithm 5.1 |
| S14 | Al-Mohy, Higham, *Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm* (MIMS EPrint 2011.83) | §6 pp. 11–12: Experiment 1 (s = 16, m = 6 for `iss_schur_new`) |
| S15 | Deadman, Higham, Ralha, *A Recursive Blocked Schur Algorithm for Computing the Matrix Square Root* (MIMS EPrint 2012.26) | title page only; the algorithm is read from its implementation S3 |
| S16 | this repository at `a9b2524`: `linalg/*.go`, `docs/linalg.md`, `random/libm.go`, `internal/kernels/expcr.go`; go-sparse `docs/design-direct.md` §4; go-numerics/special `docs/design-complex.md` | §2 |

**Not obtained / not opened** (no claim below rests on them): Higham,
*Functions of Matrices* (2008); Dhillon & Parlett (MRRR, dstemr's theory);
Higham & Tisseur 2000 (block 1-norm estimator); Higham & Lin 2011 (fractional
powers); Al-Mohy & Higham 2009 on the Fréchet derivative; Higham 1987 (real
square roots); Kågström & Poromaa (dtrsen/dtgsen reordering); Moler & Stewart
1973 (QZ); Golub & Van Loan (Parlett recurrence, `funm`). Each implementing
task opens the one it ports from before starting.

## 2. What exists (row-major, validated by tolerance)

`linalg` today re-expresses LAPACK algorithms in row-major Go, generic over
`float64 | complex128`, validated **by tolerance against numpy** (docs/linalg.md):

| internal | file:line | LAPACK counterpart | bit-exact with Reference-LAPACK? |
|---|---|---|---|
| `getrf` (recursive) | lu.go:22 | dgetrf2 | no: row-major order, packed GEMM |
| `potrf` | cholesky.go:14 | dpotrf2 | no |
| `geqrf`, `orgqr` | qr.go:40, 60 | dgeqrf/dorgqr | no |
| `gebal`, `gehrd` | eig.go:20, 183 | dgebal (pre-3.12), dgehd2 (unblocked) | no |
| `lahqr`, `zlahqr` | eig.go:337, 560 | dlahqr only — never dlaqr0 | no |
| `hetrd`, `steqr`, `stedc` | eigh.go:39, 297; dc.go:21 | dsytrd, dsteqr, dstedc (own merge) | no |
| `nrm2`, `lapy2` | blas.go:94, 122 | not dnrm2 (Blue) nor dlapy2 (`math.Hypot`) | no |
| `gemm` → `fastGemm` | blas.go:170, 206 | packed SIMD GEMM, may use FMA | no |

Three facts shape the design:

1. **Nothing is bit-exact today**, and three of the gaps are structural: the
   packed GEMM (k-blocked, FMA), the unblocked Hessenberg reduction, and
   dlahqr where dhseqr switches to dlaqr0 above NMIN = 75 (S12 dhseqr.f:451–465,
   iparmq.f:246). `Eig`'s 7× gap at n = 1024 (docs/linalg.md) is that last one.
2. **`random/libm.go` already has correctly rounded `log`, `log1p`, `exp`,
   `expm1`, `pow`** (double-double, rounded once), and the rule that every
   product feeding a sum is converted explicitly so Go cannot fuse it.
3. **Complex elementary functions** come from `math/cmplx`; the bit-faithful
   layer is go-numerics/special `cmplx`, whose design ends with a nested-module
   carve-out so that go-ndarray can import it without a module cycle
   (design-complex.md "Module boundary").

## 3. What SciPy 1.18.1 calls

| SciPy | path (S-ref) | LAPACK/BLAS reached (Reference-LAPACK names) |
|---|---|---|
| `lu_factor`, `lu_solve` | `getrf` (lu.py:119), `getrs` (212) | dgetrf → dgetrf2, dlaswp, dtrsm, dgemm; dgetrs |
| `lu` | C++ `_linalg_lu` (lu.py:350; _linalg_lu_det.hh:174) | dgetrf on a column-major copy |
| `cho_factor`, `cho_solve` | C++ `_cholesky` (potrf, uplo flipped for C order, _linalg_cholesky.hh:38–90); `potrs` | dpotrf → dpotrf2, dsyrk, dtrsm; dpotrs |
| `cholesky_banded`, `cho_solve_banded` | `pbtrf`, `pbtrs` | dpbtrf/dpbtf2, dpbtrs |
| `solve_triangular` | `trtrs` (_basic.py:379) | dtrtrs → dtrsm |
| `solve_banded` | `gtsv` if l = u = 1 (509), else `gbsv` (516) | dgtsv; dgbtrf/dgbtf2/dgbtrs |
| `solveh_banded` | `ptsv` for one off-diagonal (651), else `pbsv` (661) | dpttrf/dpttrs; dpbtrf/dpbtrs |
| `qr(pivoting=True)` | C++ `_qr` → `geqp3` (_linalg_qr.hh:197), `orgqr` (227) | dgeqp3 → dgeqrf for fixed columns, dlaqps / dlaqp2 (dgeqp3.f:254, 335, 350); dorgqr |
| `hessenberg` | `gebal(permute=0)` then `gehrd`, `orghr` (_decomp.py:1532–1548) | dgebal('N'), dgehrd (dlahr2, dlarfb; NB, NX from ilaenv), dorghr |
| `matrix_balance` | `gebal` (_basic.py:1818) | dgebal (3.12 algorithm) |
| `schur` | `gees` with `sort_t` (_decomp_schur.py:170–203); `output='complex'` casts first | dgees: dgebal('P'), dgehrd, dorghr, dhseqr, dtrsen (dgees.f:365–414); zgees |
| `rsf2csf` | Python Givens loop, `eigvals` of each 2×2 block (_decomp_schur.py:324–336) | dgeev on 2×2; numpy `dot`, `norm` |
| `qz` | `gges`, `sort_t=0` (_decomp_qz.py:122) | dgges: dggbal('P'), dgeqrf, dormqr, dorgqr, dgghrd, dhgeqz, dggbak (dgges.f:474–579) |
| `ordqz` | `gges` then `tgsen(ijob=0)` (_decomp_qz.py:435) | dtgsen → dtgexc/dtgex2 (dtgsen.f:639) |
| `eigh_tridiagonal` | `'auto'`: `stevd` for all, `stebz`+`stein` for a subset (_decomp.py:1377–1446); options `stemr`, `sterf`, `stev` | dstevd → dsterf / dstedc (dstevd.f:275–277); dstebz, dstein |
| `eig_banded`, `eigvals_banded` | `sbevd`/`hbevd` for all; `sbevx`/`hbevx`, `abstol = 2·lamch('s')` for a subset (_decomp.py:836–857) | dsbevd → dsbtrd, dsterf or dstedc + dgemm (dsbevd.f:329–340); dsbevx |
| `expm` | C `matrix_exponential_*` (S2) | dgemm, dgemv, dcopy, dscal, daxpy, dgetrf, dgetrs("T"); dlacn2 only for n ≥ 400 |
| `sqrtm` | C `matrix_squareroot_*` (S3) | dgees("V","N"), dlanv2, dtrsyl, dgemm; complex: zgemm, ztrsyl |
| `logm` | Python (S5): `schur`, `rsf2csf`, `_sqrtm_triu` (S4: Python blocks + `trsyl`), `solve_triangular`, `onenormest` (S11), `roots_legendre` (S10 → `eigvals_banded` → dsbevd), then `expm(F)` as an accuracy check | all of the above |
| `fractional_matrix_power` | Python (S5): `svdvals`, `matrix_power`, `schur`, `rsf2csf`, `_sqrtm_triu`, `solve_triangular`, `onenormest` | dgesdd, dgees, dtrsyl, dtrtrs |
| `funm`, `signm` | `schur` + `rsf2csf`, pythran `_funm_loops` (Parlett), `dot`; `signm` falls back to a Newton iteration with `svd`, `inv` | dgees, dgesdd, dgetrf/dgetri |
| `cosm`…`tanhm` | `expm(±1j·A)`; `tanm`/`tanhm` add `solve` | as `expm` (complex path) |
| `expm_frechet`, `expm_cond` | Python Padé with `lu_factor`/`lu_solve` (S6:275–276); `expm_cond` builds the n²×n² Kronecker form (S6:304–406) | dgetrf, dgetrs, dgemm |

Answers to the brief's key questions: SciPy does **not** call dstemr by
default (only `lapack_driver='stemr'`); it calls **dstevd** (all) or
**dstebz + dstein** (subset) for tridiagonal, **dsbevd**/**dsbevx** for
banded, **dgees** (with dtrsen for `sort`), **dgehrd**, **dgges**, **dtgsen**.
**dtrsen** is reached only through `gees`; SciPy has no `ordschur`, so
MATLAB's `ordschur` is a direct dtrsen/ztrsen wrapper here.

### SciPy behaviours found while reading (each gets a test that reproduces it)

- **Q1 — real `expm` adds A⁶ to U for m = 13.** `pade_UV_calc_s/_d` form
  `U = (L·M + N)·A` with `beta = 1` into the buffer that still holds A⁶
  (_matfuncs_expm.c:1294, 1513); the complex variants use `beta = 0`
  (1843, 2173). Measured with the 1.18.1 wheel against mpmath (50 digits) on
  `c·I + 0.1·randn(5)`: relative error real/complex = 5.9e-14 / 1.2e-15 (c = 3),
  6.1e-13 / 3.8e-15 (c = 4), 2.6e-12 / 2.4e-14 (c = 30). **Decision**: the Go
  port uses `beta = 0` (a documented departure); the C oracle is built with the
  same one-line patch, so tier 2 still holds; the gap is reported upstream
  (not yet filed).
- **Q2 — `trsyl`'s scale is multiplied, not divided.** dtrsyl solves
  `op(A)X ± X op(B) = scale·C` (dtrsyl.f:41), so X/scale solves the original
  system; SciPy stores `X·scale` (_matfuncs_sqrtm.c:887–895,
  _matfuncs_sqrtm.py:103). Only reached when scale < 1 (overflow guard).
  Ported as is, with a test that forces scale < 1; departure decided after
  that test shows the effect.
- **Q3 — `sqrtm` of a size-1 non-2-D input returns `exp`** (_matfuncs.py:391–392).
  Not ported: Go `Sqrtm` takes matrices only (`ErrShape` below 2-D).
- **Q4 — the triangular `expm` recomputes the superdiagonal as
  `(e^b − e^a)/(b − a)`** (_matfuncs_expm.c:2472, 2485), which cancels when
  a ≈ b. Ported as is (it is what SciPy returns); tested at a = b(1 + 2⁻⁴⁰).
- **Q5 — m-selection tests are strict** (`eta < theta`, expm.c:496) where
  Algorithm 5.1 states "≤" (S13 p. 983). Ported as SciPy has it.
- Diagonal input short-cuts to `exp` of the diagonal (expm.c:2409–2414);
  `funm`/`signm`/`logm` print or warn instead of raising — Go returns the
  estimate (`FunmResult.Err`, `LogmResult.ErrEst`) and no warning.

## 4. Exactness decision

SciPy's wheel uses OpenBLAS (architecture kernels, FMA, blocked sums, and its
own `getrf`/`potrf`), so its values cannot be reproduced bit-for-bit in
general. Following go-sparse (design-direct.md §4), by tier:

1. **Integers and structure, exact against the SciPy 1.18.1 wheel**: pivots
   (`getrf`, `gbtrf`), `jpvt` of `geqp3`, `ilo`/`ihi` and permutations of
   `gebal`/`ggbal`, `sdim` of a sorted `schur`/`ordqz`, the selected eigenvalue
   count of `stebz`/`sbevx`, `(m, s)` of `expm` (exact 1-norms for n < 400,
   S2:524), `(s, m)` of `logm` when `onenormest` is exact (n ≤ 2) or fed the
   same draws (S11:214; `random`'s legacy MT19937 reproduces `np.random`).
   Each decision compares a computed value with a threshold; the Go code
   records the smallest relative margin, and the gate **reports instead of
   failing** below 1e-12. Prerequisite: the LAPACK version inside the wheel's
   OpenBLAS is identified (task 0); where it predates 3.12 (dgebal was
   rewritten in 3.12.0), the integer gate for that routine is against the
   oracle only.
2. **Values, bit-exact against a C oracle**: SciPy 1.18.1 and NumPy built
   from source against **Reference-LAPACK v3.12.1's own BLAS and LAPACK**,
   `-O2 -ffp-contract=off` (gcc on aarch64 and clang on darwin contract by
   default), NumPy with SIMD dispatch disabled, and `exp`, `log`, `pow`,
   `hypot`, `exp2`, `log2` resolved to CORE-MATH's correctly rounded C (MIT),
   so that both sides return the unique correctly rounded value. The oracle
   carries the Q1 patch. It is built by a Go driver from user-supplied
   checkouts in a durable directory, never vendored. This tier covers every
   routine in this document: SciPy's C (S2, S3) and Python (S1, S4–S6) are
   ported **operation for operation**, including buffer layout (SciPy copies
   to column-major, so the Go side holds the same column-major buffer), the
   `lwork` SciPy passes (it selects blocked or unblocked LAPACK paths) and
   Reference `ilaenv`/`iparmq` constants.
3. **Values against the wheel, by tolerance**: backward errors and
   identities (§6), with the 32·n·ε·κ rule of docs/linalg.md.

What makes tier 2 reachable in Go:

- **No fusion.** Every `x*y + z` is written `float64(x*y) + z` (Go's spec
  lets the compiler fuse otherwise, and it does on arm64, ppc64le, s390x,
  riscv64, loong64); a test on each fusing GOARCH checks a known
  fused/unfused pair.
- **Reference loop order for BLAS.** Each routine reproduces netlib's loops:
  `dgemm` NN accumulates `C(i,j) += (alpha·B(l,j))·A(i,l)` in increasing l
  after scaling C by beta; `ddot`'s unrolling keeps left-to-right order;
  **dnrm2 is `BLAS/SRC/dnrm2.f90`** (Blue's algorithm) and `dlartg` is
  `SRC/dlartg.f90`, both as shipped in v3.12.1; `dlapy2` as in
  dlapy2.f:100–110 — not `math.Hypot`.
- **gfortran complex arithmetic**: `(a+bi)(c+di)` as `ac−bd`, `ad+bc` with
  each product rounded; complex `/` by the range-reduced (Smith) formula
  gfortran inlines under `-fcx-fortran-rules` — not Go's runtime division.
- **Correctly rounded libm**: `random/libm.go` moves to `internal/libm`
  (shared by `random` and `linalg`) and gains `exp2`, `log2`, `hypot`.
- **Single precision as SciPy**: the new `scipy.linalg`-named functions are
  generic over `float32 | float64 | complex64 | complex128` and run s/c
  routines in single precision, as SciPy does (`slamch` constants per type).
  The numpy-named functions keep numpy's widening.

**Per routine**:

| routine family | tier 1 (wheel) | tier 2 (oracle) | waits for |
|---|---|---|---|
| LU, Cholesky, triangular, banded solves, `geqp3` | pivots, `jpvt` | bit-exact | — |
| `gebal`, `gehrd`, `orghr`, `hessenberg`, `matrix_balance` | `ilo`, `ihi`, permutation | bit-exact | wheel's LAPACK version |
| `gees`/`dhseqr`/`dtrsen`, `schur`, `ordschur`, `rsf2csf` (real) | `sdim` | bit-exact | — |
| `rsf2csf`, complex `schur` (zgees) | `sdim` | bit-exact | — |
| `gges`, `tgsen`, `qz`, `ordqz` | `sdim`, `ilo`/`ihi` | bit-exact | — |
| `stevd`, `stebz`, `stein`, `sbevd`, `sbevx` | eigenvalue counts, `iblock` | bit-exact | — |
| `dstemr` (`lapack_driver='stemr'` only) | count | tolerance only until its bibliography is read | Dhillon–Parlett |
| `expm` real | `(m, s)` | bit-exact (with Q1 patch on both sides) | — |
| `expm` complex, `cosm`…`tanhm` | `(m, s)` | bit-exact except the diagonal short-cut (`cexp`) | special `cmplx` |
| `sqrtm` real without negative eigenvalues | singular/ill-conditioned flags | bit-exact | — |
| `sqrtm` complex path, `logm`, `fractional_matrix_power` | `(s, m)` with seeded draws | bit-exact for the real-arithmetic parts; `csqrt`, complex `log`, `power`, `_logm_superdiag_entry`, `_unwindk` by tolerance | special `cmplx` (then bit-exact against npymath/libm of the oracle) |
| `funm`, `signm` | — | bit-exact for the factorization and Parlett loops; user `func` is the caller's | — |
| `expm_frechet`, `expm_cond` | — | bit-exact | — |

## 5. API (Go, package `linalg`)

Stacks follow the package's rule (leading axes are a batch, one matrix per
goroutine; SciPy's `_apply_over_batch`). Errors: existing `ErrShape`,
`ErrArgs`, `ErrSingular`, `ErrNotPosDef`, `ErrNoConvergence`; new
`ErrReorder` (trsen/tgsen `info = 1`; gees `info = n+1, n+2`) and
`ErrTooLarge` (resource caps, §8). Non-finite input is refused with
`ErrLinAlg` (SciPy's `check_finite=True`).

```go
// (a) exports
func LU(a *ndarray.Array, permuteL bool) (p, l, u *ndarray.Array, err error)
func LUFactor(a *ndarray.Array) (lu, piv *ndarray.Array, err error)        // piv 0-based, as SciPy
func LUSolve(lu, piv, b *ndarray.Array, trans Trans) (*ndarray.Array, error) // Trans: NoTrans, Transpose, ConjTrans
func ChoFactor(a *ndarray.Array, lower bool) (c *ndarray.Array, err error)
func ChoSolve(c *ndarray.Array, lower bool, b *ndarray.Array) (*ndarray.Array, error)
func CholeskyBanded(ab *ndarray.Array, lower bool) (*ndarray.Array, error)
func ChoSolveBanded(cb *ndarray.Array, lower bool, b *ndarray.Array) (*ndarray.Array, error)
func SolveTriangular(a, b *ndarray.Array, o TriOpts) (*ndarray.Array, error) // Trans, Lower, UnitDiagonal
func SolveBanded(l, u int, ab, b *ndarray.Array) (*ndarray.Array, error)
func SolvehBanded(ab, b *ndarray.Array, lower bool) (*ndarray.Array, error)
func QRPivoted(a *ndarray.Array, mode QRMode) (q, r, p *ndarray.Array, err error)
func Hessenberg(a *ndarray.Array, calcQ bool) (h, q *ndarray.Array, err error)
func MatrixBalance(a *ndarray.Array, o BalanceOpts) (BalanceResult, error)  // MATLAB balance
func Schur(a *ndarray.Array, out Output, sort Select) (SchurResult, error)   // T, Z, Sdim
func RSF2CSF(t, z *ndarray.Array) (tc, zc *ndarray.Array, err error)
func OrdSchur(t, z *ndarray.Array, sel []bool) (SchurResult, error)          // MATLAB ordschur (dtrsen)

// (b) matrix functions
func Expm(a *ndarray.Array) (*ndarray.Array, error)
func Cosm, Sinm, Tanm, Coshm, Sinhm, Tanhm(a *ndarray.Array) (*ndarray.Array, error)
func Sqrtm(a *ndarray.Array) (SqrtmResult, error)                 // X, Singular, IllConditioned
func Logm(a *ndarray.Array, o NormEstOpts) (LogmResult, error)     // X, ErrEst; o.Rand = draw source
func FractionalMatrixPower(a *ndarray.Array, t float64, o NormEstOpts) (*ndarray.Array, error)
func Funm(a *ndarray.Array, f func(diag []complex128) []complex128) (FunmResult, error) // X, Err
func Signm(a *ndarray.Array) (*ndarray.Array, error)
func ExpmFrechet(a, e *ndarray.Array, m FrechetMethod) (expA, l *ndarray.Array, err error)
func ExpmCond(a *ndarray.Array) (float64, error)
func OneNormEst(op LinearOperator, o NormEstOpts) (NormEstResult, error) // shared with go-sparse

// (c) tridiagonal and banded
func EighTridiagonal(d, e *ndarray.Array, o EigSelect) (w, v *ndarray.Array, err error)
func EigvalshTridiagonal(d, e *ndarray.Array, o EigSelect) (*ndarray.Array, error)
func EigBanded(ab *ndarray.Array, lower bool, o EigSelect) (w, v *ndarray.Array, err error)
func EigvalsBanded(ab *ndarray.Array, lower bool, o EigSelect) (*ndarray.Array, error)
// EigSelect: All | ByValue(vl, vu) | ByIndex(il, iu), Driver (Auto, Stevd, Stebz, Sterf, Stev, Stemr), Tol, MaxEV

// (d) generalized Schur
func QZ(a, b *ndarray.Array, out Output) (QZResult, error)                  // AA, BB, Q, Z
func OrdQZ(a, b *ndarray.Array, sort Select, out Output) (OrdQZResult, error) // + Alpha, Beta
```

`Select` is `SortNone`, `LHP`, `RHP`, `IUC`, `OUC` (SciPy's predicates, S7,
including `ordqz`'s β = 0 rules) or `SelectFunc(func(re, im float64) bool)`;
QZ predicates receive `(alpha, beta)`. `NormEstOpts.Rand` defaults to a
legacy MT19937 seeded with 0, so `Logm` is deterministic (SciPy's is not:
unseeded global `np.random`). `LinearOperator` is a two-method interface
(`Matvec`, `Rmatvec` on blocks) so go-sparse reuses the one `onenormest`.

## 6. Validation plan

- **Harness in Go** (`internal/cmd/linalgoracle`): writes inputs as `.npy`,
  runs the wheel (`uv run --with scipy==1.18.1 python -I`) and the C oracle
  with a dispatcher string held in Go, reads `.npy` back; fixtures stored with
  hex floats. No Python file in the repository for new fixtures.
- **Corpus per routine**: 0×0, 1×1, 2×2 (every dlanv2/dlag2 branch), orders
  below and above every switch point (NMIN = 75 and NTINY = 15 of dhseqr,
  NX = 128 of dgehrd (ilaenv.f:624), 25 of ilaenv's ISPEC 9 (ilaenv.f:715,
  dstedc's leaf size), n = 400 of `expm`, SciPy's
  block size 64 of `_sqrtm_triu`), each `(m, s)` of `expm` and `logm`,
  triangular and diagonal inputs (short-cuts), real negative eigenvalues
  (`sqrtm`/`logm` complex path), exactly singular, defective (Jordan),
  graded, Hilbert, Grcar, Frank, clustered and repeated eigenvalues (stein
  reorthogonalization, stedc deflation), banded l/u from 0 to n−1, every
  `Select` predicate with β = 0 and α = β = 0 pairs, float32 and complex64.
- **Tier 1** gates on the wheel: integers equal (margin rule), identities
  `‖A − ZTZᴴ‖/‖A‖`, `‖ZᴴZ − I‖`, `‖Q(A,B)Zᴴ − …‖`, `‖X² − A‖/‖A‖` for `sqrtm`,
  `‖e^F − A‖₁/‖A‖₁` for `logm`, Sturm counts for tridiagonal eigenvalues,
  backward errors for solves — within 32·n·ε·κ.
- **Tier 2** gates on the oracle: every output bit-equal; a failure prints the
  first differing element and both values.
- **Ground truth** (mpmath, 50 digits) for matrix functions: the Q1 table is
  the first case; S14's Experiment 1 matrix checks `(s, m) = (16, 6)`.
- **SciPy's own tests** for these functions are ported case by case (each
  task opens the corresponding `scipy/linalg/tests/` file).
- **Negative controls**: drop one `float64()` conversion on arm64 → the tier-2
  gate fails; restore Q1's `beta = 1` → the mpmath gate fails; force
  `ErrNoConvergence` in each iteration (dhseqr, dhgeqz, dstein, dstedc).
- **Gauss rules**: `EigvalsBanded` reproduces `roots_legendre(m)` for m ≤ 64
  bit-for-bit against the oracle (S10 adds one Newton step in NumPy, ported
  in special, not here).

## 7. Performance plan

Reference: the SciPy 1.18.1 wheel with `OPENBLAS_NUM_THREADS=1`, one core,
interleaved rounds, never at 1-minute load above ~8 (deferred and said so).
Rows: `expm`, `logm`, `sqrtm`, `schur`, `hessenberg`, `qz`, `ordqz`,
`eigh_tridiagonal`, `eig_banded`, `solve_banded`, `lu_factor`,
`qr(pivoting)` at n = 16, 64, 256, 1024 (QZ to 512); banded n = 10⁵ with
kd = 1, 5, 50; allocations counted exactly.

1. Exact port first (it is the oracle); measure; a slower row is an open
   defect.
2. **Algorithm first**: the exact ports already bring dlaqr0 (multishift QR
   with aggressive early deflation) and blocked dgehrd — the two causes of
   `Eig`'s 7× gap — so `Eig` migrates onto the exact dgeev path (task 9).
3. **Kernels**: one `refgemm` micro-kernel in `internal/kernels` (go-asmgen),
   register-blocked over i and j, k sequential and increasing, multiply then
   add without FMA, `alpha·B(l,j)` formed first, beta applied as netlib does —
   bit-identical to netlib `dgemm` while SIMD. `dtrsm`, `dtrmm`, `dsyrk`,
   `dgemv` (the other hot reference routines) vectorize over the independent
   index the same way. Expected cost of no FMA: about half the peak of an FMA
   GEMM; the table decides.
4. Fused or k-reordered GEMM is allowed only for routines validated by
   tolerance; none is planned; any exception is named in docs/linalg.md.
5. Stacks of small matrices run in parallel (existing `parallelFor`); single
   matrices stay single-threaded for the comparison.

## 8. Security and resource bounds

- **Allocation before arithmetic**: every size product (`n²`, `6n² + 4n` of
  `expm`'s workspace, `(2l+u+1)·n` of `gbsv`, `n²·n²` of `expm_cond`) is
  computed with overflow checks for 32-bit GOARCH (386, arm); `ExpmCond`
  refuses n above a documented cap (`ErrTooLarge`; n = 100 already needs
  800 MB) and bomb tests assert the refusal under `GOMEMLIMIT` and
  `ulimit -v`.
- **Untrusted indices**: `LUSolve`'s `piv`, `OrdSchur`'s `sel`, banded `l`, `u`,
  `EigSelect` bounds (`il > iu`, `vl >= vu`, NaN bounds) are validated before
  any LAPACK-style loop indexes with them.
- **Termination**: LAPACK iterations keep their caps (dhseqr, dhgeqz, dstein,
  dstedc → `ErrNoConvergence`); `logm`'s square-root loop and `signm`'s Newton
  loop get explicit caps; non-finite input is refused up front, which removes
  the NaN-driven endless loops of `_inverse_squaring_helper`.
- **Callbacks**: `SelectFunc` and `Funm`'s `f` are called a bounded number of
  times; a wrong-length result from `f` is `ErrArgs`, not a panic.
- **Fuzzing** (corpus committed under `testdata/fuzz`): `FuzzSchur`,
  `FuzzOrdSchur`, `FuzzQZ`, `FuzzOrdQZ`, `FuzzExpm`, `FuzzSqrtm`, `FuzzLogm`,
  `FuzzBandedSolve`, `FuzzTridiagSelect`, `FuzzLUSolvePiv`: no panic, no hang
  (time budget), error or identity within bound.
- **govulncheck** is not in this repository's CI yet; task 1 adds it.

## 9. File layout

```text
internal/libm/            correctly rounded log, log1p, exp, expm1, pow (moved from random/), + exp2, log2, hypot
internal/kernels/         refgemm micro-kernels (go-asmgen, every GOARCH), no FMA
linalg/internal/lapack/   exact column-major ports of Reference-LAPACK v3.12.1, generic over 4 types
  refblas.go              level 1-3 BLAS in netlib loop order (dnrm2 Blue, ddot, daxpy, dgemv, dger, dtrmv, dtrsv, dgemm, dtrmm, dtrsm, dsyrk)
  arith.go                no-fuse helpers, gfortran complex mul/div, lamch, ilaenv, iparmq
  getrf.go potrf.go trtrs.go banded.go (gbtrf gbtf2 gbtrs gtsv pttrf pttrs pbtrf pbtf2 pbtrs)
  geqp3.go (geqrf geqr2 laqps laqp2 orgqr org2r larft larfb larfg larf)
  gebal.go gehrd.go (lahr2 gehd2 orghr) hseqr.go (lahqr laqr0-5 lanv2) trsen.go (trexc laexc lasy2 trsyl laln2 lacn2) gees.go
  sbtrd.go sterf.go stedc.go (laed0-9 lamrg) stevd.go sbevd.go stebz.go stein.go sbevx.go
  gges.go (ggbal gghrd hgeqz ggbak) tgsen.go (tgexc tgex2 tgsyl tgsy2 lag2)
linalg/decomp_scipy.go    LU, LUFactor, LUSolve, ChoFactor, ChoSolve, banded, SolveTriangular, QRPivoted
linalg/schur.go           Schur, RSF2CSF, OrdSchur, Hessenberg, MatrixBalance
linalg/tridiag.go         EighTridiagonal, EigBanded, ...
linalg/qz.go              QZ, OrdQZ
linalg/expm.go            port of _matfuncs_expm.c; Cosm ... Tanhm
linalg/sqrtm.go           port of _matfuncs_sqrtm.c and _sqrtm_triu
linalg/logm.go            port of _matfuncs_inv_ssq.py
linalg/funm.go frechet.go onenormest.go
linalg/testdata/scipy/    fixtures (hex floats), generated by internal/cmd/linalgoracle
internal/cmd/linalgoracle Go harness (wheel + oracle)
internal/cmd/refscipy     Go driver that builds the oracle from user-supplied checkouts in a durable directory
```

The existing row-major internals stay until their exact replacement lands
and is measured; then each numpy-named function moves onto the exact layer in
its own PR (one implementation per algorithm), its numpy fixture still
passing by tolerance.

## 10. Ordered tasks (one PR each)

0. **Oracle and harness**: `refscipy` driver (Reference-LAPACK v3.12.1 BLAS +
   LAPACK, SciPy 1.18.1 + NumPy from source, `-ffp-contract=off`, CORE-MATH
   libm, Q1 patch); `linalgoracle` harness; identify the LAPACK version in the
   wheel's OpenBLAS; fixture format.
1. **`internal/libm`** (move from `random`, add `exp2`, `log2`, `hypot`),
   no-fuse helpers and the fused/unfused test on every fusing GOARCH;
   govulncheck in CI.
2. **Reference BLAS and LAPACK auxiliaries** (`refblas.go`, `arith.go`,
   Householder and scaling auxiliaries) with the `refgemm` kernel; bit-exact
   against netlib.
3. **LU, Cholesky, triangular exports**: `LU`, `LUFactor`, `LUSolve`,
   `ChoFactor`, `ChoSolve`, `SolveTriangular`.
4. **Banded solves and banded Cholesky**: `SolveBanded`, `SolvehBanded`,
   `CholeskyBanded`, `ChoSolveBanded`.
5. **Pivoted QR**: `QRPivoted` (dgeqp3 exact).
6. **Hessenberg and balance**: `Hessenberg`, `MatrixBalance`.
7. **Schur**: dhseqr with dlaqr0–5, dgees with dtrsen, z-variants; `Schur`,
   `RSF2CSF`, `OrdSchur`.
8. **Tridiagonal and banded eigensolvers**: `EighTridiagonal`,
   `EigvalshTridiagonal`, `EigBanded`, `EigvalsBanded` (stevd, stebz, stein,
   sbevd, sbevx); Gauss–Legendre check.
9. **`Eig` onto the exact dgeev path** (gebal 'B', dhseqr, dtrevc3): closes the
   7× gap; numpy fixture by tolerance.
10. **`Expm`** and `Cosm`…`Tanhm` (port of _matfuncs_expm.c, Q1 departure).
11. **`Sqrtm`** (C recursion and the blocked `_sqrtm_triu`; Q2 test).
12. **`OneNormEst`** (port of `_onenormest.py`, legacy MT19937 draws,
    `LinearOperator`).
13. **`Logm`, `FractionalMatrixPower`** (port of `_matfuncs_inv_ssq.py`).
14. **`Funm`, `Signm`**.
15. **`ExpmFrechet`, `ExpmCond`** (size cap, bomb test).
16. **`QZ`** (dgges with dggbal, dgghrd, dhgeqz, dggbak; z-variants).
17. **`OrdQZ`** (dtgsen `ijob = 0`, dtgexc, dtgex2, dtgsyl).
18. **Performance pass**: the §7 table in docs/linalg.md, a defect per slower
    row.
19. **Complex bit parity**: switch the complex scalar functions to special
    `cmplx` once its nested module is released; upgrade the §4 rows that wait
    for it.
20. **`stemr` driver option**, after reading Dhillon–Parlett.

Each task updates docs/linalg.md, docs/coverage.md, the README, the
go-ndarray landing page and the docs site, with 100% coverage and its fuzz
targets.
