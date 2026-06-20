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
OpenBLAS/BLIS structure). With the by-lane-FMLA micro-kernel it reaches **parity
with tuned multi-threaded OpenBLAS 0.3.29 at n=1024 (≈0.99×, ~203 GFLOP/s)** and
~0.97× at n=512; only small n=256 still trails (per-call overhead). See the
matmul section below.

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

`go ×` is NumPy ÷ go-ndarray: **> 1 means go-ndarray wins.** Two go forms are
shown for the binary/sqrt ops:

- **alloc** — `a + b` / `np.sqrt(a)`: returns a fresh array (NumPy `x + y`).
- **into** — `a.AddInto(out, b)` / `a.SqrtInto(out)`: writes a caller-supplied
  buffer, the no-allocation form (NumPy `np.add(x, y, out=z)` / `np.sqrt(x, out=z)`).
  This is the apples-to-apples small-N comparison: it removes the per-op result
  allocation, which is the *only* thing NumPy beats us on at small n (see below).

| op | n | NumPy 2.2.4 | go alloc | go into | go × alloc | go × into |
|------|------:|------:|------:|------:|:--:|:--:|
| Add  | 1 024     |     432 |  1 049 |   **148** | 0.41 (numpy) | **2.41** |
| Add  | 4 194 304 | 1 600 974 | **768 722** | **429 410** | **2.08** | **2.10** |
| Mul  | 1 024     |     424 |    978 |   **168** | 0.43 (numpy) | **2.08** |
| Mul  | 4 194 304 | 1 572 361 | **763 339** | **435 109** | **2.06** | **1.81** |
| Sqrt | 1 024     |     496 |  1 299 |   **263** | 0.38 (numpy) | **1.80** |
| Sqrt | 4 194 304 | 1 669 901 | **605 561** | **333 289** | **2.76** | **3.22** |
| Sum  | 1 024     |     637 |  **130** |       — | **4.90** | — |
| Sum  | 4 194 304 |   496 000 | **193 679** |     — | **2.56** | — |
| Max  | 1 024     |     611 |  **160** |       — | **3.83** | — |
| Max  | 4 194 304 |   326 466 | **231 626** |     — | **1.41** | — |

(All numbers measured back-to-back in one VM session, 2026-06; go is
`-benchtime=1s`, NumPy `min`-of-runs in `docs/bench_numpy.py`. Add/Sub/Mul/Div
now run a SIMD kernel — packed SSE2 `ADD/SUB/MUL/DIVPD` on amd64, NEON `VFMLA`
on arm64 — so even the *alloc* form wins ~2× at large n. Sum/Max are reductions
with no result array, so they have no separate *into* row.)

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
  per-arch and bit-for-bit against NumPy's `np.sqrt`). arm64 now also has a
  **packed NEON `FSQRT V.2D`** kernel (two doubles per instruction, emitted by
  `WORD` since Go's assembler has no vector-sqrt mnemonic), replacing the prior
  one-lane-at-a-time scalar `FSQRTD` loop. Small `n=1 024`: the *alloc* form loses
  to NumPy on allocation cost (as Add/Mul), but `a.SqrtInto(out)` wins 1.8×.
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
- **Small arrays (n≈1 024) for Add/Mul/Sqrt — the *alloc* form: NumPy wins; the
  *into* form: go-ndarray wins ~2×.** The entire gap is the result allocation.
  `make([]float64, 1024)` measures ~900–1 080 ns on this VM's allocator (it zeroes
  the 8 KiB the kernel then fully overwrites), while NumPy serves the temp from a
  cached free-list for ~150 ns. So `a + b` at n=1024 is allocator-bound (≈1 µs,
  of which the SIMD compute is only ~150 ns) and loses 0.4×; **`a.AddInto(out, b)`
  — the no-allocation form, NumPy's `np.add(x, y, out=z)` — removes that cost and
  wins 1.8–2.4× at the same size.** This is a Go-allocator ceiling, not a kernel
  one: Go's `make` always zeroes and offers no unsafe "give me raw bytes" escape,
  and returning pooled memory would break the caller-owns-the-result contract, so
  the allocating small-N form cannot match NumPy's free-list — the `*Into` parity
  path is the supported way to hit/beat NumPy there (and at large n even the alloc
  form wins, because the kernel time then dwarfs the one allocation). (Sum and Max are
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

All rows measured **back-to-back, interleaved, in the same VM session** (the only
fair way on a shared host whose load drifts), 4 samples each, `go ÷ OB` reported
as the per-sample ratio (robust to the absolute time drifting between sessions).

| n (N×N) | go-ndarray (4 core) | OpenBLAS-mt | go GFLOP/s | OB GFLOP/s | go ÷ OB |
|------:|------:|------:|------:|------:|:--:|
| 256  |    261 000 |    174 000 | 102.8 | 154.2 | 0.67× |
| 512  |  1 530 000 |  1 500 000 | 175.4 | 178.9 | 0.97× |
| 1024 | 10 600 000 | 10 580 000 | 202.6 | 203.0 | **≈0.99× (parity; samples 0.96–1.00)** |

**Verdict: at n=1024 the pure-Go (+ generated asm) GEMM reaches PARITY with tuned
OpenBLAS** (~0.99×, individual samples straddling 1.00×), sustaining **~203
GFLOP/s** — up from the prior kernel's ~23 GFLOP/s and from this kernel's own 0.76×
before the by-lane-FMLA fix below. n=512 is ~0.97×; only small n=256, where fixed
per-call overhead (panel packing + the 4-way goroutine fan-out) dominates the
~250 µs of compute, still trails at 0.67× — a small-matrix overhead ceiling, not a
kernel-throughput one (the same kernel hits parity once the matrix is large enough
to amortise the setup).

### The fix that reached parity: the by-lane FMLA

The single change that took n=1024 from **0.76× to ≈0.99×** was the arm64
micro-kernel's inner FMA. The kernel broadcasts one packed A value into every
column of its row; the prior version did this with `MOVD` (A double → GP register)
+ `VDUP` (GP → all lanes of a vector) before a plain vector `VFMLA` — an extra
instruction *and* a GP↔SIMD register-file round-trip per A value, every k-step.
The fix uses the indexed-element double FMLA `FMLA Vd.2D, Vn.2D, Vm.D[i]` that
every tuned arm64 dgemm uses: load the four packed A doubles of a k-step as two
`D2` vectors and have each row's FMA read its A scalar straight from a lane — no
GP detour, no `VDUP`. Go's arm64 assembler cannot *name* this indexed form
("illegal combination … ELEM"), so it is emitted by its raw 32-bit `WORD`
encoding (see `fmlaElem` in `asmgen/arm64/gen.go`, cross-checked against
`objdump`); this is still pure Go assembly, CGO=0.

Measured on this Apple-silicon core: the L1-resident micro-kernel went **~44 →
~58 GFLOP/s/core (~1.28×)**, and the full GEMM tracked it to parity. (An earlier
note on this page claimed the indexed form was *slower* on Apple silicon — that
was wrong; it had never actually been measured against a correct encoding.)

Two further levers were prototyped and **measured to NOT help on this core**, so
they were not shipped (documented here so they are not re-attempted blindly):
- **Software pipelining** (prefetch + double-buffer the next k-step's A/B into
  shadow registers, rotate after the FMAs): ~2% on the L1 micro-kernel but **1–9%
  *slower* on the full GEMM** — the M-series out-of-order window already hides the
  L1/L2 load latency, and the extra `VMOV` rotations + register pressure cost more
  than they save.
- **Wider 6×8 tile** (24 accumulators vs 16): same GFLOP/s as the 4×8 — the loop
  is FMA-latency-bound, not register-pressure-bound, so a wider tile adds no
  throughput.
- **MC/KC/NC retuning**: a 12-point sweep confirmed the shipped 256/256/512 is
  already the optimum on this core.

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

### Where the remaining gap is (small n only)

At n=1024 there is effectively **no** kernel-throughput gap left: the by-lane
FMLA inner loop runs ~58 GFLOP/s/core, the full GEMM hits ~203 GFLOP/s, and
go ÷ OB straddles 1.00×. The only sizes still short are **small** matrices, and
there the gap is *fixed per-call overhead*, not compute:

- At n=256 the matmul is ~250 µs of real work, but every call still pays the full
  panel-pack pass and a 4-way goroutine fan-out. OpenBLAS amortises that setup
  far better (smaller pack cost, a thread pool that is already warm), so it leads
  0.67× here even though the per-core FMA rate is the same. As n grows the setup
  is amortised away — n=512 is already 0.97×.

This is a small-matrix overhead ceiling. Closing it further would mean a
serial-vs-parallel crossover tuned per size and a cheaper pack for tiny panels;
it is not a Go-assembler or algorithm limit, and it does not affect the
large-matrix parity result. (`GOAMD64=v3`/AVX2-FMA would likewise lift the amd64
micro-kernel, which today is SSE2-only; the arm64 path already has its FMA.)

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
- **Add / Sub / Mul / Div** — the elementwise SIMD kernels are held
  **bit-identical** to the scalar oracle across every length residue mod 8 and the
  IEEE edge inputs (signed zeros, ±Inf, NaN, extremes): each lane is a single
  independent correctly-rounded IEEE op (no reduction grouping), and on arm64 the
  FMA-against-an-exact-constant form (`b+a·1`, `a−b·1`, `0+a·b`) rounds identically
  to the plain op, so bit-identity is the contract.
- **Sum** — validated to a tight *relative tolerance* (not bit-identity): its
  lane-parallel grouping is a valid reordering, the same kind NumPy's pairwise
  summation uses, so closeness — not bit-identity — is the contract for a
  floating-point reduction.

## Where go-ndarray still loses, and the plan

| op | status | note |
|----|--------|-----|
| `Add`/`Sub`/`Mul`/`Div`/`Sqrt`, **alloc** form, small n (≈1 K) | NumPy wins (Go-allocator ceiling) | the result `make` (~900 ns zeroing 8 KiB) is the whole cost; Go has no unzeroed-alloc escape and pooling would break result ownership. The `*Into` no-alloc form **wins 1.8–2.4×** — the supported parity path. Large-n alloc form already wins ~2×. |
| `Add`/`Sub`/`Mul`/`Div` | **FIXED — go wins ~2× large n, *Into* wins all n** | new SIMD kernels: packed SSE2 `ADD/SUB/MUL/DIVPD` (amd64), NEON `VFMLA`/`VFMLS` (arm64); bit-identical to the scalar oracle |
| `Sqrt` | **FIXED — go wins 2.8× at 4 M, *Into* 1.8× small** | packed `SQRTPD` (amd64) / packed NEON `FSQRT V.2D` (arm64) / intrinsic `FSQRTD` (others), off a non-`func`-pointer seam |
| `Max` / `Min` | **FIXED — go wins 1.4× at 4 M, ~4× small** | builtin-`max` NaN-propagating oracle + 4-accumulator reducer + amd64 `MAXPD`+NaN-scan |
| `MatMul` vs tuned BLAS (OpenBLAS) | **FIXED — parity at n=1024 (~0.99×, ~203 GFLOP/s); 0.97× at n=512** | by-lane FMLA micro-kernel (`FMLA Vd.2D,Vn.2D,Vm.D[i]` via `WORD`) closed the prior 0.76× gap. Only small n=256 trails (0.67×) on per-call overhead, not throughput — see above |
| other `Map` ufuncs (`Exp`, `Log`, `Sin`…) | NumPy ~parity (libm-bound) | a packed `VEXP`/`VLOG` is libm-accuracy work; the math, not the dispatch, dominates here |

## SIMD coverage

- **amd64 (SSE2)** ships hand-vectorized `sum` (4-accumulator `ADDPD`), `sqrt`
  (packed `SQRTPD`), `max`/`min` (`MAXPD`/`MINPD` + `CMPPD` NaN scan), the
  **elementwise `add`/`sub`/`mul`/`div`** (packed `ADD/SUB/MUL/DIVPD`, 8
  doubles/iter + scalar tail), and the **GEMM micro-kernel** (`gemmMicro4x4`: a
  4×4 SSE2 `MULPD`+`ADDPD` tile, no FMA at the v1 baseline), generated by
  go-asmgen and validated per-arch in CI (and cross-run under qemu-x86_64 — the
  GEMM tile included).
- **arm64 (NEON)** ships hand-vectorized `sum`, **packed `sqrt`** (`FSQRT V.2D`),
  the **elementwise `add`/`sub`/`mul`** (via `VFMLA`/`VFMLS` against a `1.0`
  vector — `add: b+a·1`, `sub: a−b·1`, `mul: 0+a·b`, each FMA exact so
  bit-identical to the plain op; `div` stays on scalar `FDIVD`, no vector form),
  and the **GEMM micro-kernel** (`gemmMicro4x8`: a 4×8 tile, 16 D2 accumulators,
  using the **by-lane FMLA** `FMLA Vd.2D, Vn.2D, Vm.D[i]`). Three instructions are
  emitted by raw `WORD` because Go's arm64 assembler cannot name them: the vector
  `FSQRT V.2D`, and the indexed-element `FMLA …D[i]` (both verified vs `objdump`).
  `max`/`min` use the four-accumulator builtin-`max`/`min` (lowering to
  `FMAXD`/`FMIND`); the C-accumulate fold still uses `VFMLA` against `1.0` (no
  plain vector FP add exists). All beat/parity NumPy via SIMD + multicore, no
  `func`-pointer indirection.
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
