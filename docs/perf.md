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

NumPy's `matmul` is BLAS, a different fight: go-ndarray answers it with its own
**panel-packed, cache-blocked GEMM** + go-asmgen SIMD-FMA micro-kernel (the
OpenBLAS/BLIS structure). It does not beat *tuned* OpenBLAS but reaches ~76% of
it and is 3–7× over the prior kernel — see the matmul section below.

## Test bench

- **Machine**: debian arm64 (Apple-silicon Tart VM), 4 vCPU.
- **NumPy**: 2.2.4. Elementwise/reduction rows use Debian **reference BLAS
  3.12.1** (irrelevant to those ufuncs). The matmul rows are measured against
  **OpenBLAS 0.3.29 (pthread, multi-threaded)** — the tuned BLAS — selected via
  `update-alternatives --set libblas.so.3 …/openblas-pthread/libblas.so.3` and
  confirmed loaded by reading `/proc/self/maps` (libopenblasp-r0.3.29.so). This
  is the meaningful bar: the earlier reference-BLAS matmul comparison is kept only
  as a footnote.
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

## Matrix multiply vs TUNED BLAS (OpenBLAS) — the hard frontier

This is the honest fight: go-ndarray's pure-Go (+ generated asm) GEMM against
**OpenBLAS 0.3.29 pthread**, decades-tuned multi-threaded assembly. Both run on
the same 4-vCPU Apple-silicon VM, back-to-back. GFLOP/s = `2·n³ / time`.

| n (N×N) | go-ndarray (4 core) | go GFLOP/s | OpenBLAS-mt | OB GFLOP/s | go ÷ OB |
|------:|------:|------:|------:|------:|:--:|
| 64   |     19 828 |  26.4 |     11 548 |  45.4 | 0.58× |
| 128  |     73 627 |  57.0 |     24 660 | 170.1 | 0.33× |
| 256  |    328 575 | 102.1 |    172 787 | 194.2 | 0.53× |
| 512  |  1 999 951 | 134.2 |  1 348 483 | 199.1 | 0.67× |
| 1024 | 13 753 750 | 156.1 | 10 506 282 | 204.4 | **0.76×** |

**Verdict: a real, big step forward, but pure Go does NOT beat tuned OpenBLAS.**
The packed GEMM sustains **~156 GFLOP/s at n=1024 (≈76% of OpenBLAS)**, up from
the prior kernel's ~23 GFLOP/s, but OpenBLAS's hand-scheduled asm still leads at
every size. This is shipped because it is **3–7× faster than the prior kernel**
(below), not because it wins the OpenBLAS fight — it does not, and that is stated.

### What shipped: a panel-packed, cache-blocked GEMM

The earlier register-blocked attempt read A and B straight from the source
matrices; on power-of-two row strides its destination rows collided in L1 and it
lost to the autovectorized ikj kernel at large n, so it was capped to `n ≤ 224`
and **packing was prototyped but not shipped** (the previous note here). This
round implements the full OpenBLAS/BLIS structure and ships it for **all** sizes:

- **Packing** — A is copied into MR-tall row panels and B into NR-wide column
  panels in contiguous, unit-stride scratch (pooled, allocation-free per call).
  The micro-kernel then streams conflict-free memory **regardless of the source
  stride** — this is the precise fix for the power-of-two L1 set-conflicts that
  defeated the unpacked attempt. Edge tiles are zero-padded so the kernel always
  sees a full MR×NR block; the ragged right/bottom borders take a scalar path.
- **Cache blocking** — an `NC` (columns) → `KC` (contraction) → `MC` (rows) loop
  nest keeps the packed B panel L2-resident and each packed A panel in L1.
  Defaults `MC=256, KC=256, NC=512` (tuned on this VM).
- **SIMD-FMA micro-kernel** (go-asmgen) — **NEON 4×8 on arm64** (16 D2
  accumulators, the source of the win), **SSE2 4×4 on amd64** (8 XMM
  accumulators; SSE2 has no FMA, so explicit MULPD+ADDPD), **scalar 4×4** on the
  four arches without vector-double asm — which still gain from contiguous packed
  data + blocking + multicore.
- **Parallelism** — the M rows are split into MR-aligned bands, one per core,
  each writing a disjoint region of C.

The arithmetic is the standard ikj order `dst[i][j] += a[i][p]·b[p][j]`; packing
only relocates the operands, so the result is **bit-for-bit identical** to the
scalar oracle and to NumPy's `A@B` (max abs diff **0.0**, verified at 128×128
against OpenBLAS).

### Packed GEMM vs the PRIOR kernel (why it ships)

Measured back-to-back in one binary under identical load (so the ratio is robust
to the noisy VM); the prior kernel is the register-blocked/ikj path this replaces:

| n | prior kernel | packed | speedup |
|------:|------:|------:|:--:|
| 64   |  6–9 GF |  20–21 GF | **2.2–3.4×** |
| 128  | 16 GF   |  32–52 GF | **3.1–4.2×** |
| 256  | 10–19 GF|  53–119 GF| **5.0–6.2×** |
| 512  | 13–21 GF|  80–156 GF| **6.0–7.3×** |
| 1024 | 16–22 GF|  99–156 GF| **6.2–7.2×** |

The packed kernel wins at **every** size, so it is shipped uniformly — no
size-routing — and the old `n ≤ 224` blocked/ikj split is removed.

### The ceiling: why pure Go can't match OpenBLAS's asm

The pure-Go+asm FMA *throughput* is not the wall — a hand-written NEON-FMLA loop
measured **57 GFLOP/s/core** here, *above* OpenBLAS's 48 GFLOP/s/core
single-thread, so the SIMD math is there. The gap (≈24% at n=1024) comes from
micro-kernel scheduling that Go's arm64 assembler **cannot express**:

1. **No by-lane FMLA.** Every tuned arm64 dgemm uses
   `FMLA Vd.2d, Vn.2d, Vm.d[i]` to broadcast one A lane straight into the FMA.
   Go's assembler rejects the indexed-element form (`illegal combination … ELEM`),
   so each A value must be moved to a GP register and `VDUP`'d into a vector
   before a plain vector `VFMLA` — an extra instruction per A value in the inner
   loop, and a vector-register-file round-trip OpenBLAS does not pay.
2. **No plain vector FP add.** Go's arm64 backend exposes only `VFMLA`/`VFMLS`
   for vector double, so even the closing `C += acc` is done as `C + acc·1.0` via
   FMLA against a constant `1.0` vector (bit-exact, but a wasted multiply).
3. **No software pipelining / prefetch scheduling.** OpenBLAS interleaves loads
   of the next k-step with the current FMAs and issues `PRFM` prefetches; the
   asm here is straight-line and relies on the core's OoO window, which does not
   fully hide L2 latency at the larger panels — visible as the 134→156 GFLOP/s
   climb only at n=1024.
4. **Narrower micro-kernel.** 32 V-registers cap a NEON kernel at a 4×8 (or 8×4)
   tile once operands need registers; OpenBLAS uses an 8×8-class tile with
   careful register renaming, giving a higher compute-to-load ratio.

None of these are pure-Go *algorithm* limits — they are Go-assembler expressivity
limits. Closing the last ~24% would need either those instructions exposed by the
Go arm64 assembler (by-lane FMLA in particular) or `GOAMD64=v3`/AVX2-FMA on
x86. The result stands as the realistic pure-Go ceiling on this micro-arch:
**~76% of tuned OpenBLAS, 3–7× over the prior kernel.**

### Footnote: vs reference BLAS

Against Debian's single-threaded **reference BLAS 3.12.1** (the prior comparison
on this page), the packed GEMM wins by a wide margin at every size — but that is
not a meaningful bar for a matmul claim, so the table above uses OpenBLAS.

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
| `MatMul` vs prior kernel | **FIXED — 3–7× faster, all sizes** | panel-packed cache-blocked GEMM + go-asmgen SIMD-FMA micro-kernel (NEON 4×8 / SSE2 4×4); shipped uniformly (done) |
| `MatMul` vs tuned BLAS (OpenBLAS) | go reaches ~76% of OpenBLAS at n=1024; OpenBLAS still leads | the remaining ~24% is Go-assembler expressivity (no by-lane FMLA / no vector FP add / no SW-pipelining), not a pure-Go algorithm limit — see the ceiling analysis above |
| other `Map` ufuncs (`Exp`, `Log`, `Sin`…) | NumPy ~parity (libm-bound) | a packed `VEXP`/`VLOG` is libm-accuracy work; the math, not the dispatch, dominates here |

## SIMD coverage

- **amd64 (SSE2)** ships hand-vectorized `sum` (4-accumulator `ADDPD`), `sqrt`
  (packed `SQRTPD`), `max`/`min` (`MAXPD`/`MINPD` + `CMPPD` NaN scan), and the
  **GEMM micro-kernel** (`gemmMicro4x4`: a 4×4 SSE2 `MULPD`+`ADDPD` tile, no FMA
  at the v1 baseline) kernels, generated by go-asmgen and validated per-arch in
  CI (and cross-run under qemu-x86_64 — the GEMM tile included).
- **arm64 (NEON)** ships the hand-vectorized `sum` kernel and the **GEMM
  micro-kernel** (`gemmMicro4x8`: a 4×8 NEON `VFMLA` tile, 16 D2 accumulators).
  Both work around the same two arm64-assembler limits: no plain vector-double add
  (so accumulation/store fold uses `VFMLA` against a `1.0` vector) and no by-lane
  `FMLA` (so each A value is `VDUP`'d into a vector before the vector `VFMLA`). It
  has **no packed vector-double sqrt/max/min** (only `VFMLA`/`VFMLS` exist), so
  `sqrt` uses the compiler's scalar `FSQRTD` intrinsic and `max`/`min` the
  four-accumulator builtin-`max`/`min` (which lowers to `FMAXD`/`FMIND`) — both
  beat NumPy via the intrinsics + multicore, no `func`-pointer indirection.
- The other four 64-bit Go targets — **riscv64, loong64, ppc64le, s390x** — keep
  the validated scalar oracles (Go's loong64/ppc64le assemblers expose no
  vector-double arithmetic; riscv64's V extension is optional), using the same
  four-accumulator max/min, direct sqrt loop, and a **scalar 4×4 GEMM
  micro-kernel** over the packed panels, and still get the **packing + cache
  blocking + multicore** win, so they also beat single-threaded NumPy on large
  arrays. (s390x additionally exercises the big-endian path in CI.)

All six are exercised in CI (native amd64/arm64 + qemu for the rest); each
per-arch job regenerates the committed `.s`, fails if it is stale, vets
(asmdecl), builds (cmd/asm encodes), and runs the bit/NaN-correctness suite. The
multicore path and the packed/cache-blocked GEMM driver are
architecture-independent; only the GEMM micro-kernel is per-arch (NEON 4×8 on
arm64, SSE2 4×4 on amd64, scalar 4×4 on the other four — all bit-identical to the
scalar ikj oracle, validated per-arch in CI incl. amd64 under qemu-x86_64).
