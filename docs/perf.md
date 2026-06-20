# Performance — go-ndarray vs NumPy

Honest, reproducible head-to-head of `go-ndarray/ndarray` against **NumPy
2.2.4** on identical hardware. "On n'a pas le droit de se tromper": every number
here is measured, every win is real, and where NumPy still leads it says so.

## How a pure-Go library can beat NumPy

NumPy's elementwise and reduction ufuncs (`+`, `*`, `sqrt`, `sum`, `max`, …) are
**single-threaded SIMD C**. So a pure-Go library wins them by combining two
levers NumPy does not use for these ops:

1. **Multicore** — fan the contiguous loop across `GOMAXPROCS` goroutines above a
   size threshold (small arrays stay serial so scheduling never dominates).
2. **SIMD via [go-asmgen](https://github.com/go-asmgen)** — for the sum
   reduction, whose scalar `s += v` loop the Go compiler will not auto-vectorize
   (it would change FP grouping), a hand-written NEON/SSE2 kernel with four
   independent accumulators recovers the lane parallelism NumPy's C has.

A third lever — **eliminating allocation** — turned out to be the single biggest
factor: the old elementwise path materialised two full broadcast copies *and* a
destination per op (3×N). The same-shape contiguous fast path now passes the
operands' live backing slices to the kernel and allocates only the result.

NumPy's `matmul` is BLAS, so it is a different fight (see the matmul note below).

## Test bench

- **Machine**: debian arm64 (Apple-silicon Tart VM), 4 vCPU.
- **NumPy**: 2.2.4, linked against Debian **reference BLAS 3.12.1**
  (single-threaded; *not* OpenBLAS — see the matmul caveat).
- **Go**: 1.26.4, `CGO_ENABLED=0`, `GOWORK=off`.
- **go-ndarray bench**: `go test -bench=. -benchtime=1s` (`bench_test.go`),
  ns/op. Multi-core = default `GOMAXPROCS=4`; single-core = `GOMAXPROCS=1`.
- **NumPy bench**: `docs/bench_numpy.py` (matched ops/sizes), min-of-7 ns/op.

Reproduce: rsync the repo to the VM, then
`go test -bench=. -benchtime=1s .` and `python3 docs/bench_numpy.py`.

## Elementwise and reductions (ns/op; lower is better)

`go ×` is NumPy ÷ go-ndarray (multi-core): **> 1 means go-ndarray wins.**

| op | n | NumPy 2.2.4 | go (1 core) | go (4 core) | go × (4 core) |
|------|------:|------:|------:|------:|:--:|
| Add  | 1 024 | 438 | 952 | 1 461 | **0.30** (numpy) |
| Add  | 4 194 304 | 1 696 681 | 1 728 300 | **824 689** | **2.06** |
| Mul  | 1 024 | 458 | 919 | 1 443 | 0.32 (numpy) |
| Mul  | 4 194 304 | 1 646 489 | 1 495 747 | **787 378** | **2.09** |
| Sqrt | 1 024 | 476 | 1 173 | 1 658 | 0.29 (numpy) |
| Sqrt | 4 194 304 | 1 637 880 | 2 363 166 | **934 570** | **1.75** |
| Sum  | 1 024 | 671 | **135** | **135** | **4.95** |
| Sum  | 4 194 304 | 505 997 | **493 755** | **209 908** | **2.41** |
| Max  | 1 024 | 628 | **159** | **157** | **4.00** |
| Max  | 4 194 304 | 323 062 | 621 257 | **241 269** | **1.34** |

(Sqrt/Max rows re-measured 2026-06 after the SIMD-kernel round; Add/Mul/Sum
unchanged from the previous round. Each go number is the best of three
`-benchtime=1s` runs; each NumPy number the min of seven — taken back-to-back in
the same VM session so they share the machine's instantaneous load.)

### Reading the table

- **Add / Mul on large arrays: go-ndarray wins ~2×.** Allocation elimination +
  multicore. Note even single-core large Mul edges out NumPy now (the kernel is
  memory-bandwidth bound and the old copies were the bottleneck).
- **Sum: go-ndarray wins everywhere — up to ~5× on small, 2.4× on large.** The
  SIMD 4-accumulator kernel already beats NumPy single-core (493 µs vs 506 µs at
  4 M); multicore then roughly halves it again. Small sums are dominated by
  NumPy's per-call Python/dispatch overhead, which Go does not pay.
- **Sqrt: go-ndarray now WINS 1.75× at 4 M (was a ~1.3× loss).** The fix was to
  stop routing `Sqrt` through `Map`'s `func(float64) float64`: that indirection
  blocked *both* the Go compiler's `FSQRTD` intrinsic (arm64) and a packed SSE2
  `SQRTPD` (amd64), making the old path pay a real function call *per element*
  (~2.7× slower single-core). A dedicated `sqrt` kernel behind the kernels seam —
  packed `SQRTPD` on amd64, the intrinsic-lowered scalar `FSQRTD` loop on
  arm64/others — plus the existing multicore fan-out is what turns the loss into
  a win. The kernel is **bit-identical** to a scalar `math.Sqrt` loop (validated
  per-arch and bit-for-bit against NumPy's `np.sqrt`). Small `n=1 024` still goes
  to NumPy: below the parallel threshold a single core is sqrt-throughput-bound
  and the per-call `*Array` allocation dominates — the accepted small-size
  trade-off (same as Add/Mul).
- **Max: go-ndarray now WINS — 4.0× small, 1.34× at 4 M (was 0.62 / 0.31
  losses).** Two changes. (1) The scalar oracle was switched from `if v > m`
  (which silently *ignored* NaNs, diverging from NumPy) to the **builtin
  `max`/`min`**, which is NaN-propagating *and* lowers to the hardware `FMAXD`/
  `FMIND` intrinsic — `math.Max` would also propagate NaN but is an
  un-intrinsified call ~12× slower in this hot loop. (2) The SIMD max/min
  reduction uses **four independent accumulators** to break the dependency chain
  (~3.6× over one), exactly like the sum kernel: packed `MAXPD`/`MINPD` + a NaN
  scan on amd64, register-unrolled builtin `max`/`min` on arm64/others. See the
  NaN convention note below.
- **Small arrays (n≈1 024) for Add/Mul/Sqrt: NumPy still wins.** Below the
  parallel threshold go-ndarray is serial and its per-call `*Array` allocation
  costs more than NumPy's tight C loop at this size. This is the right
  trade-off: parallelising a 1 K array would *slow it down*. (Sum and now Max are
  the exceptions — their reduction kernels need no result allocation and win even
  serially.)

### NaN convention for Max / Min (and the SIMD max kernel)

`Max`/`Min` are **NaN-propagating**: if any element is NaN the result is NaN.
This matches `numpy.max`/`numpy.min`, Go's builtin `max`/`min`, and IEEE-754
*maximum* (not the C `fmax` / IEEE `maxNum` "ignore NaN" rule); on signed zeros
it returns `+0` for `max(-0,+0)` and `-0` for `min(-0,+0)`, also matching NumPy.
The previous `if v > m` scalar form silently skipped NaNs (and a *leading* NaN
poisoned the scan), so it disagreed with NumPy; this is the corrected, documented
semantics, and the entire stack — scalar oracle, the four-accumulator reducer,
the amd64 `MAXPD`/`MINPD` kernel, and the multicore fold — is held **bit-for-bit
identical** to it (NaN at any position → NaN; exact extreme otherwise), validated
across every length residue and NaN placement in CI. The amd64 kernel gets this
despite `MAXPD`'s own non-propagating NaN rule by carrying a parallel
`CMPPD`-unordered OR-mask and forcing the result to NaN if any lane ever held
one — so the value `MAXPD` leaves in a NaN lane is irrelevant.

## Matrix multiply (ns/op; lower is better)

| n (N×N) | NumPy 2.2.4 (ref BLAS) | go (1 core) | go (4 core) | go × (4 core) |
|------:|------:|------:|------:|:--:|
| 64  | 71 721 | **44 828** | **46 334** | **1.55** |
| 128 | 633 613 | 366 270 | **154 917** | **4.09** |
| 256 | 5 125 533 | 6 167 394 | **1 669 947** | **3.07** |
| 512 | 36 970 300 | 43 531 510 | **10 967 031** | **3.37** |

- **go-ndarray's GEMM now wins at every measured size, 1.55–4.09×** against this
  NumPy's BLAS. The new lever is a **4-row register-blocked micro-kernel**: it
  accumulates four output rows at once so each loaded `b[p][j]` is fused into all
  four rows, quartering the b-traffic and loop overhead. At `n=64` this makes
  even the **single core beat reference BLAS** (44.8 µs vs 71.7 µs, 1.60×) — the
  old scalar ikj single-core *lost* here (0.78× multicore). The block is gated to
  `n ≤ blockMaxN` (224); above it the four destination rows (n·8 bytes apart)
  start colliding in L1 on power-of-two widths, so MatMul switches to the
  autovectorized ikj kernel — whose single contiguous AXPY inner loop the Go
  compiler vectorizes cleanly — which carries the 256/512 wins via multicore.
  Both paths compute the identical ikj summation order, so the result is
  **bit-for-bit identical** (verified zero-diff against `A @ A` in NumPy).
- **Why not register-block the large case too:** on this Apple-silicon arm64 the
  four-row kernel is *slower* than the autovectorized ikj for `n ≥ 256` — the
  power-of-two row stride causes L1 set-conflict misses, and the Go compiler
  vectorizes the single-stream ikj AXPY better than the four-stream block. Fixing
  the large case the way a tuned BLAS does (A/B panel **packing** into contiguous
  scratch + an explicit SIMD-FMA micro-kernel) was prototyped and measured here;
  in pure Go it did not beat the compiler's autovectorized ikj on this micro-
  architecture, so it is not shipped. The honest scope stands: blocking wins the
  small/medium sizes now, and the large sizes win via autovectorized-ikj +
  multicore.
- **Honest caveat (unchanged):** this NumPy links Debian's **reference BLAS**,
  single-threaded and unoptimized. Against **OpenBLAS**/**MKL** (cache-blocked,
  hand-tuned, multi-threaded), NumPy's matmul would be several times faster and
  would likely lead at the large sizes. The wins above are real on this
  configuration and stated with their scope.

## Correctness

Every result is validated **bit-for-bit against NumPy** (shapes and values), not
just timed — see the cross-check in the work log: `Sqrt` (bit-identical to
`np.sqrt` via a `uint64` view), `Max`, `Min` (exact, incl. NaN propagation),
`Add`, `Mul` and the full `MatMul` (zero max-abs-diff against `A @ A`) match
NumPy at n = 100 000 / 80×80.

The SIMD kernels are also validated against the scalar oracle per-arch in CI:

- **Sqrt** — `SQRTPD` (amd64) is held **bit-identical** to a scalar `math.Sqrt`
  loop across every length residue mod 8 and the IEEE edge cases (negatives →
  NaN, ±Inf, signed zeros, explicit NaN); sqrt is a single correctly-rounded
  operation, so bit-identity (not mere closeness) is the contract.
- **Max / Min** — the four-accumulator reducer and the amd64 `MAXPD`/`MINPD`
  kernel are held **bit-identical** to the builtin-`max`/`min` oracle across
  every length and with a NaN injected at the start, middle, end, and as the sole
  element (all must return NaN); ±Inf and signed zeros bit-match too.
- **Sum** — validated to a tight *relative tolerance* (not bit-identity): its
  lane-parallel grouping is a valid reordering, the same kind NumPy's pairwise
  summation uses, so closeness — not bit-identity — is the contract for a
  floating-point reduction.

## Where go-ndarray still loses, and the plan

| op | status | note |
|----|--------|-----|
| small arrays (n≲a few K) for Add/Mul/Sqrt | NumPy wins (no parallelism, per-`*Array` alloc) | lower threshold / pooling is not worth the small-size risk; accepted |
| `Sqrt` | **FIXED — go wins 1.75× at 4 M** | dedicated `SQRTPD`/intrinsic-`FSQRTD` kernel off a non-`func`-pointer seam (done) |
| `Max` / `Min` | **FIXED — go wins 1.34× at 4 M, 4× small** | builtin-`max` NaN-propagating oracle + 4-accumulator reducer + amd64 `MAXPD`+NaN-scan (done) |
| `MatMul` small/medium | **FIXED — go wins 1.55–4.1×** | 4-row register-blocked micro-kernel (done) |
| `MatMul` large vs tuned BLAS | reference-BLAS: go wins 3.4×; OpenBLAS/MKL: BLAS would lead | A/B-panel packing + SIMD-FMA micro-kernel did not beat autovectorized ikj in pure Go on this µ-arch; not shipped |
| other `Map` ufuncs (`Exp`, `Log`, `Sin`…) | NumPy ~parity (libm-bound) | a packed `VEXP`/`VLOG` is libm-accuracy work; the math, not the dispatch, dominates here |

## SIMD coverage

- **amd64 (SSE2)** ships hand-vectorized `sum` (4-accumulator `ADDPD`), `sqrt`
  (packed `SQRTPD`), and `max`/`min` (`MAXPD`/`MINPD` + `CMPPD` NaN scan)
  kernels, generated by go-asmgen and validated per-arch in CI (and cross-run
  under qemu-x86_64).
- **arm64 (NEON)** ships the hand-vectorized `sum` kernel (`VFMLA` against a
  vector of 1.0 — Go's arm64 assembler exposes no plain vector-double add). It
  has **no packed vector-double sqrt/max/min** (only `VFMLA`/`VFMLS` exist), so
  `sqrt` uses the compiler's scalar `FSQRTD` intrinsic and `max`/`min` the
  four-accumulator builtin-`max`/`min` (which lowers to `FMAXD`/`FMIND`) — both
  beat NumPy via the intrinsics + multicore, no `func`-pointer indirection.
- The other four 64-bit Go targets — **riscv64, loong64, ppc64le, s390x** — keep
  the validated scalar oracles (Go's loong64/ppc64le assemblers expose no
  vector-double arithmetic; riscv64's V extension is optional), using the same
  four-accumulator max/min and direct sqrt loop, and still get the **multicore**
  win, so they also beat single-threaded NumPy on large arrays.

All six are exercised in CI (native amd64/arm64 + qemu for the rest); each
per-arch job regenerates the committed `.s`, fails if it is stale, vets
(asmdecl), builds (cmd/asm encodes), and runs the bit/NaN-correctness suite. The
multicore path and the register-blocked GEMM are architecture-independent.
