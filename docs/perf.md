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
  **packed NEON `FSQRT V.2D`** kernel (two doubles per instruction; emitted by
  `WORD` until v0.5.2, `VFSQRT` since Go 1.27 names it), replacing the prior
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
  serially.) **Superseded for loops by `Workspace`, see below.**

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
  sees a full MR×NR block; a ragged right/bottom tile runs the same micro-kernel
  into a scratch tile, and only its valid corner is added to C.
- **Cache blocking** — an `NC` (columns) → `KC` (contraction) → `MC` (rows) loop
  nest keeps the packed B panel L2-resident and each packed A panel in L1.
  Defaults `MC=256, KC=256, NC=512` (tuned on this VM).
- **SIMD-FMA micro-kernel** (go-asmgen) — **NEON 4×8 on arm64** (16 D2
  accumulators, the source of the win), **AVX2/FMA 6×8 on amd64** (12 YMM
  accumulators, picked at run time by a CPUID probe; an SSE2 6×4 pair without
  FMA is the fallback), **scalar 4×4** on the
  four arches without vector-double asm — which still gain from contiguous packed
  data + blocking + multicore.
- **Parallelism** — the M rows are split into MR-aligned bands, one per core,
  each writing a disjoint region of C.

The arithmetic is the ikj order `dst[i][j] += a[i][p]·b[p][j]`, summed per KC
block; for k ≤ KC that is the scalar oracle's order exactly, and above it the
block partials regroup the sum (a valid reordering, as in every blocked BLAS).
At 128×128 the result matched NumPy's `A@B` against OpenBLAS with max abs diff
**0.0**.

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
large-matrix parity result. (amd64 now has its AVX2/FMA kernel too; see the
real-hardware section below.)

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
| `Max` / `Min` | **FIXED — go wins 1.4× at 4 M, ~4× small** | builtin-`max` NaN-propagating oracle + 4-accumulator reducer + amd64 `MAXPD`+NaN-scan; NEON `FMAX`/`FMIN` on arm64 since v0.5.2 |
| `MatMul` vs tuned BLAS (OpenBLAS) | **FIXED — parity at n=1024 (~0.99×, ~203 GFLOP/s); 0.97× at n=512** | by-lane FMLA micro-kernel (`FMLA Vd.2D,Vn.2D,Vm.D[i]` via `WORD`) closed the prior 0.76× gap. Only small n=256 trails (0.67×) on per-call overhead, not throughput — see above |
| `Exp`, `Log`, `Log10` | **FIXED in v0.2.0** — own kernels | ports of Arm's optimized-routines, 2–3.5× `math.Exp`/`math.Log`; see the Exp and Log sections below |
| `Sin`, `Cos`, `Tan`, … (through `Map`) | ~parity with NumPy | per element on one Zen 3 core: Go `math.Sin` 10.5 ns vs NumPy 10.4, `math.Cos` 11.1 vs 10.7, `math.Tanh` 16.5 vs 13.5 (2026-10-04); `Map`'s call per element adds 15–20% |

## Allocation: `Workspace` (2026-10-04)

On real amd64 hardware the allocating forms were far worse than on the M4: on
a Zen 3 (16 cores), `x + y` lost to NumPy 0.13–0.25× at every size up to 256 Ki
elements, while `AddInto` won. Slicing copies, `Concatenate` and broadcasting
adds (whose results also allocate) lost 5–9×. Measured causes, on M4:

- **The garbage collector.** A benchmark's live heap is tiny, so a 128 KiB
  result per call triggered a GC cycle every ~22 calls (912 cycles in 20 000).
  With 256 MiB–2 GiB of live heap, which is closer to a real program, the
  overhead halves: 8.3 → 3.6 µs at 16 Ki, against 2.4 µs for `AddInto`.
- **The rest:** `make` zeroes memory the kernel overwrites anyway, and fresh
  memory is cold or page-faulted. `GOGC=off` does not help: it trades the
  cycles for page faults (9.4 µs).

A library must not set `GOGC`, and Go has no unzeroed allocation. What NumPy's
reference counting gets for free is scoped reuse. ND4J, the Java ndarray
library, has the same problem and answers it with *workspaces*: an arena that
results of one pass come from, recycled at the end of the pass. `Workspace` is
that. `ws.Use(x)` binds an input, every result computed from a bound array is
carved from the arena (64-byte aligned, capped so an `append` cannot spill),
results stay bound so a chain stays in it, and `Reset` recycles everything,
folding a pass that needed several blocks into one block. Sites whose kernel
writes every element (elementwise, ufuncs, copies, broadcasts, concatenation,
`Where`, `Clip`, scans, `Take`, `Outer`, mat·vec) skip the zeroing. Those that
accumulate (GEMM, vec·mat, axis reductions) clear their slot.

Zen 3, 16 cores pinned, NumPy 2.5.3 single-threaded, best of two runs:

| op | n | heap | **Workspace** | `*Into` | NumPy | Workspace vs NumPy |
|----|--:|--:|--:|--:|--:|:--:|
| Add | 1 024 | 2.6 µs | **0.38 µs** | 0.31 µs | 0.82 µs | 2.1× |
| Add | 16 Ki | 35 µs | **4.6 µs** | 4.3 µs | 5.1 µs | 1.1× |
| Add | 256 Ki | 498 µs | **67 µs** | 75 µs | 150 µs | 2.2× |
| Add | 4 Mi | 8.0 ms | **2.2 ms** | 2.2 ms | 6.3 ms | 2.9× |
| sqrt(x·y+x) | 1 024 | 12.2 µs | **2.6 µs** | — | 3.6 µs | 1.4× |
| sqrt(x·y+x) | 16 Ki | 112 µs | **33 µs** | — | 34 µs | 1.0× |
| sqrt(x·y+x) | 256 Ki | 1.18 ms | **0.39 ms** | — | 2.97 ms | 7.5× |
| sqrt(x·y+x) | 4 Mi | 23.4 ms | **9.1 ms** | — | 19.2 ms | 2.1× |

On M4 the same chain gains ×5 at 1 024 and ×1.6 at 4 Mi. Correctness: every
operation is checked bit for bit against its heap result, including on passes
whose recycled memory was first filled with NaN (a site that skipped zeroing it
needed would leak the NaN; planting that bug in the GEMM made the test fail).

## amd64 on real hardware (2026-10-04)

Every figure above was taken on Apple silicon. This round measured amd64 for
the first time, on an AMD EPYC 7773X (Zen 3, GCC Compile Farm cfarm421), with
go-ndarray and NumPy both pinned to the same 16 cores of one NUMA node
(`taskset -c 16-31`; Go's GOMAXPROCS follows the affinity mask). NumPy 2.5.3
with its bundled OpenBLAS 0.3.34 (DYNAMIC_ARCH, Haswell kernels on Zen 3),
once with `OPENBLAS_NUM_THREADS=1` and once with 16. go rows are the minimum
of three runs interleaved with the previous version, so drift on a shared host
hits both alike.

| op | before | after | gain | vs OpenBLAS 1 thread | vs OpenBLAS 16 threads |
|----|------:|------:|:--:|:--:|:--:|
| MatMul 1024² | 21.87 ms | **8.05 ms** | ×2.72 | 5.93× | 0.60× |
| MatMul 512² | 3.69 ms | 1.56 ms | ×2.36 | 3.89× | 0.66× |
| MatMul 256² | 0.661 ms | 0.558 ms | ×1.19 | 1.36× | 0.30× |
| MatMul 128² | 0.210 ms | 0.141 ms | ×1.49 | 0.73× | 0.36× |
| MatMul 1024×256·256×1024 | 5.72 ms | 2.71 ms | ×2.11 | 4.55× | 0.54× |
| Inner 512² (a·bᵀ) | 5.87 ms | **1.47 ms** | ×4.00 | 4.19× | **0.96×** |

Four changes, each measured on its own before the next:

1. **AVX2/FMA 6×8 micro-kernel** (the BLIS Haswell shape). In L1 it runs at
   43 GFLOP/s on one Cascade Lake core (16 flops per cycle at about 2.7 GHz,
   if the virtual machine's unknown clock is near that); the SSE2 4×4 it
   replaces had no FMA and half the vector width.
2. **Edge tiles through the micro-kernel.** With MR=6, every 256-row MC block
   ends in a 4-row edge panel, and the scalar edge loop took **40%** of one
   profile covering serial 64² and 512² products. A partial tile now runs the
   kernel into a scratch tile.
3. **Prefetch** of the C tile at kernel entry and of the A panel eight steps
   ahead: serial 768² went from 27.5 to 34.7 GFLOP/s (+26%). 90% of the time was
   already in the kernel, which ran at 28 GFLOP/s there against 43 with its
   panels in L1, so it was waiting on memory. A sweep of MC (48–252) and KC
   (128–512) moved nothing (±3%), so the blocking is unchanged.
4. **B packed by all workers.** MatMulP packs the shared B block once per
   (jc, pc) round, and one goroutine did it while the others waited: +30–70%
   in parallel. Below 64 Ki elements the launch costs more than the copy (M4:
   −8–15% at 96³–128²), so small blocks are still packed by one.

`Inner` gained the most because it used to materialise bᵀ before multiplying.
The GEMM now reads any 2-D view through its strides while packing (the copy it
makes anyway), so a transposed, sliced, reversed or broadcast operand costs no
extra pass.

**Still behind OpenBLAS with 16 threads** (0.54–0.66× on large products, 0.3×
below 256²). Where the parallel loss is, measured on the Zen 3:

- It is not the hardware: 16 *independent* serial 512² GEMMs, one per core,
  each ran at 41 GFLOP/s (656 in total), while the parallel 1024² GEMM ran its
  kernel at 15–17 GFLOP/s per core. Per-core efficiency already drops from 40
  to ~30 GFLOP/s at 2 cores.
- ~~It is not scheduling overhead: 83% of the parallel profile is the
  micro-kernel itself, so the kernel runs slower when cores share work.~~
  **Wrong, corrected the same day:** a profile of cycles samples only the cores
  that run, never the ones that wait. `perf stat` shows 7.1 of 16 CPUs busy
  during a 1024² product: the loss was mostly waiting, and the fix is in
  [Parallel GEMM: start once, wait on work](#parallel-gemm-start-once-wait-on-work-2026-10-04).
  Why the kernel itself also runs slower in parallel is still **not established**. The shared packed B panel is the suspect (on this part
  the last-level cache is split between core complexes, so the panel is partly
  remote), but the drop already shows at 2 cores of one complex, which that
  explanation does not cover.
- Taller row bands (more reuse of each B micro-panel) do not help: interleaved
  over 5 runs they were worse at 1024² and equal at 2048².
- **A 2-D tiling of C with private packing** (no barrier, nothing shared) was
  built and measured, and is **not shipped**: ×1.24 at 1024² on the Zen 3, but
  ×0.62 at 512² and Inner, and ×0.70–0.93 at every size on M4, whose cluster
  L2 holds the shared panel well. A tile repacks (rows + cols)·k elements for
  rows·cols·k multiply-adds, and a strided pack costs far more per element than
  an FMA, so tiles under ~256×256 lose.

The hardware counters were read the same day; see the next-but-one
section. An AVX-512 kernel for the hosts that have it remains open.

Correctness was checked on real hardware for five of the six 64-bit targets:
amd64 (Zen 3, FMA path), arm64 (cfarm185), ppc64le (cfarm120), riscv64 (cfarm94)
and loong64 (cfarm401). The s390x host (LinuxONE) did not answer; s390x stays
covered by the qemu lane in CI.

## Axis reductions (2026-10-04)

On the Zen 3 the axis reductions of a 1024×1024 matrix lost to NumPy 0.23–0.43×.
Two causes, both measured:

- **An axis-0 reduction ran on one core.** It has a single outer slab, and the
  driver only split outer slabs; the comment justifying that said a column split
  would need a strided gather. It does not: a band of columns is a contiguous run
  of every row. `RunAxisP` now splits the columns into bands of at least 512
  when there are fewer slabs than workers, and each row run is added with the
  elementwise SIMD kernel (exact per element, so the sum is still in axis order).
- **A row reduction (axis 1, `inner` = 1) was a sequential scalar loop**,
  latency-bound at one add per 4 cycles. Each row now goes through the SIMD
  sum/max/min. Max and min are exact; the sum is regrouped lane-parallel, as
  NumPy's own pairwise sum is.

Zen 3, 16 cores pinned, minimum of three interleaved runs:

| op (1024×1024) | before | after | NumPy 1 thread | after vs NumPy |
|----|--:|--:|--:|:--:|
| SumAxis(0) | 696 µs | **186 µs** | 180 µs | 0.97× |
| SumAxis(1) | 531 µs | **92 µs** | 224 µs | 2.4× |
| MaxAxis(1) | 601 µs | **93 µs** | 189 µs | 2.0× |

On M4: MaxAxis(1) ×6–8, SumAxis(0) ×1.9.
## Broadcasting a row without materialising it (2026-10-04)

`M + row` (1024×1024 plus 1×1024) lost to NumPy 0.26× on the Zen 3 even with a
`Workspace`: the broadcast path copied the row out to a full 8 MiB matrix
before adding. When one operand has the full shape and the other, leading 1s
dropped, is a suffix of it (a row, a 2-D mean subtracted from a 3-D stack), the
second simply repeats in blocks of its own length, so `binOp` now streams the
live slices block by block through the SIMD kernel (`kernels.RepeatP`), split
across cores. Below 64 elements per block the old path is kept.

| BroadcastAdd 1024²+row, Zen 3 | heap | `Workspace` | NumPy 1 thread |
|----|--:|--:|--:|
| before | 3.1 ms | 1.76 ms | 460 µs |
| after | **0.84 ms** | **114–196 µs** | 460 µs |
## Exp (2026-10-04)

`Exp` went through `Map(math.Exp)`, and `math.Exp` itself was the cost: 13.2 ns
per element on one Zen 3 core, 3.9 ns on M4 (the func-pointer call added only
15–20%). It is now a port of the double-precision exp of Arm's
optimized-routines (Szabolcs Nagy; glibc's exp since 2.28; MIT licence): x =
k·ln2/128 + r, 2^(k/128) from a 128-entry table as scale·(1+tail), exp(r)−1 from
a degree-5 polynomial. The common case is written out in the loop, because the
function is too large to inline and a call per element cost a third of the
time.

- **Accuracy:** worst 0.504 ULP over 80 000 inputs across the whole finite
  range, the region near 0, the overflow edge and the subnormal range,
  measured against a 300-bit reference (`math/big`, ln 2 from its atanh
  series), against Arm's documented 0.509; the committed test samples 20 000
  of them and finds 0.501. `math.Exp` measured 0.879 ULP on the 80 000 inputs
  on arm64 (0.841 on the test's 20 000). The table was not trusted as copied:
  `TestExpTable` re-derives all 256 entries from 2^(k/128) at 300 bits.
- **A Go bug it removes**, reported as
  [golang/go#81995](https://github.com/golang/go/issues/81995): on amd64,
  `math.Exp` returns **+Inf for 709.436139303104 ≤ x < 709.782712893384**,
  although the result is finite there (NumPy, glibc and mpmath:
  `exp(709.5)` = 1.3549863193146328e+308). Its assembly rounds k = x·log2(e) to
  1024 from x = 1023.5·ln 2, and the biased-exponent check treats that as
  overflow although the reduced factor is below 1. Reproduced with Go 1.26.8 and
  1.27.1 on AMD Zen 3, Intel Cascade Lake and Intel Haswell, with and without
  the FMA path; the assembly is unchanged on master. go-ndarray's `Exp`
  inherited it until v0.2.0; `TestExpTopOfRange` pins the fix.
- Just above ln(2^-1075), exp rounds up to the smallest subnormal, as glibc
  and NumPy do. Go's arm64 `math.Exp` returns 0 there (amd64 does not); that is
  one unit of the last subnormal place, not a contract violation.

Speed, kernel only, 16 Ki elements: M4 30 µs (`math.Exp` 65 µs); one Zen 3 core
62 µs (`math.Exp` 215 µs, NumPy single-threaded 78 µs). Whole `Exp` on Zen 3,
16 cores: 1 Ki ×2.7, 16 Ki ×3, 256 Ki ×1.6, 4 Mi ×1.65; from 256 Ki on it is
2–5× NumPy. A SIMD version (AVX2 gathers for the table) is the next step.
## Dot and mat·vec (2026-10-04)

`dotRange`, under `Dot` (1-D), `MatVec` and `Dot1DP`, was a four-chain scalar
loop (gc does not vectorise it): on one Zen 3 core 2^20 elements took 464 µs,
2× NumPy's single thread. Two generated kernels replace it: `dotFMA` on amd64
(four YMM accumulators, b read straight from memory by VFMADD231PD; two loads
per FMA make it load-bound, which four chains cover), gated on the FMA probe,
and `dotNEON` on arm64 (eight D2 accumulators for the four FP pipes of an Apple
core). The other targets keep `dotRange`. The lane-parallel sum is a regrouping,
held to the n·ε·Σ|aᵢbᵢ| bound and exact on integer data.

| Zen 3 | before | after | NumPy 1 thread |
|----|--:|--:|--:|
| Dot 2^20, 1 core | 464 µs | **229 µs** | 231 µs |
| MatVec 1024², 1 core | 482 µs | **134 µs** | 153 µs |
| Dot 2^20, 16 cores | 159 µs | 99 µs | — |
| MatVec 1024², 16 cores | 165 µs | 89 µs | — |

NumPy's multi-threaded dot still wins on this part (24 µs). The go-ndarray
version stops scaling at about 90 GB/s from 4 cores on. ~~A likely cause, not
verified, is cache affinity: OpenMP keeps each thread on one core, so it
rereads its own chunk from its own L3 slice on every repetition, while
goroutines move between core complexes.~~

**Measured 2026-10-07 (POWER9 bare metal, cfarm29, one thread per core):**
the cache-affinity guess does not hold. The pattern is the fork/join of
`parallelFor`, the same as the GEMM's (see *Parallel GEMM: start once, wait on
work*). A 2^20 dot took 212 µs on 4 cores and 553 µs on 8, close to the 688 µs
of one core. With 8 workers, only 2.4 CPUs were busy (`perf stat`). Three
replacements for `parallelFor` were built and measured against it
(8–10 interleaved rounds); **none is shipped**:

| scheduler | 8 workers | 4 workers |
|---|---|---|
| shared queue of 4w blocks, caller works | Dot 1.55×, MatVec 2.0×, SumAxis(1) 2.0× | Sum 4 Mi 0.51×, Dot 0.55× |
| owned runs (the static chunks) + stealing, caller works | Dot 2.18×, MatVec 2.0× | Sum 4 Mi 0.63×, Dot 0.68× |
| owned runs + stealing, caller blocks, last block wakes it | mostly 1.0–1.4×, Exp 1.87× | Sum and AddInto at 256 Ki 0.51–0.55× |

The second row keeps exactly the static chunks of the original and still
loses at 4 workers, so affinity is not the cause. `perf stat` places the loss
in scheduling instead: with the caller working, 2.1 of 4 CPUs were busy
against 3.4 for the original. Its goroutines wait in the run queue of the
caller's processor until another one is woken to steal them. With the caller
blocking, the 4-worker case recovers, but the 8-worker gains go with it. The
code is on the `experiment/parallel-for-scheduling` branch.

**Solved in v0.6.0 with a persistent helper pool.** The cost being paid was
waking threads, so the fix is to stop putting them to sleep between back-to-back
operations, as OpenBLAS and OpenMP do. Up to `GOMAXPROCS-1` helper goroutines
are started once. After an operation they poll for the next one for 200 µs,
then park. The work keeps owned runs with stealing (row two of the table
above). Reductions now use four partials per worker: with one each, a helper
that saw the job late had its only block taken by the caller, who then summed
two in a row (Sum of 4 Mi on 4 cores: 2.8 of 4 CPUs busy). On POWER9 bare metal,
each case in a **fresh process**, 6 interleaved rounds, against v0.5.2
(median):

| | 2 workers | 4 workers | 8 workers | 8 workers, before → after |
|---|---|---|---|---|
| Dot 2^20 | 3.29× | 2.41× | **10.8×** | 625 → 58 µs |
| MatVec 1024² | 2.01× | 1.65× | 5.78× | 323 → 56 µs |
| SumAxis(1) | 2.06× | 1.80× | 7.09× | 310 → 44 µs |
| Sum 256 Ki | 2.33× | 3.23× | 5.60× | 61 → 11 µs |
| Max 256 Ki | 1.75× | 1.34× | 6.78× | 757 → 112 µs |
| AddInto 256 Ki | 2.22× | 1.91× | 5.57× | 102 → 18 µs |
| Exp 256 Ki | 1.18× | 1.49× | 1.97× | 1013 → 513 µs |
| Sum 4 Mi | 1.33× | 1.17× | 1.41× | 431 → 305 µs |
| Exp 4 Mi | 1.00× | 1.10× (p25 0.84×) | 1.29× | |
| BroadcastAdd | 0.95× | 1.01× | 1.67× | |
| Chain (Mul, Add, Sqrt) 4 Mi | | 0.91× | 1.04× | |
| MatMul 256², 512² (own driver) | | 1.00× | 0.96–0.99× | |

A first combined run, with every benchmark in one process, also showed serial
operations speeding up (Sqrt of 1 Ki, 1.5–2.2×). Run alone, those are equal.
That gain belonged to whatever state the earlier benchmarks had left, not to
this change, so only fresh-process figures are reported here. The cost: after
an operation, the helpers spend up to 200 µs of CPU polling. They are never
stopped (see the README, *Goroutines*). Not yet measured on amd64 or on Apple
M, whose hosts were unreachable or loaded at the time.

## Log and Log10 (2026-10-04)

`Log` went through `Map(math.Log)`: 11.6 ns per element on a Zen 3 core
against NumPy's 4.2. It is now a port of the double-precision log of Arm's
optimized-routines (the companion of `Exp`, also glibc's since 2.28; MIT
licence): x = 2^k·z, z split into 128 subintervals, log(x) = k·ln2 + log(c) +
log1p(z/c − 1) with 1/c and log(c) from a table and a degree-6 polynomial;
inputs within 2⁻⁴ of 1 take a degree-12 polynomial. z/c − 1 is one
`math.FMA` (exact on every target), so the second table the non-FMA variant
needs is not used. `Log10` is log(x)·(1/ln10), the formula of `math.Log10`, on
this log.

- **Accuracy:** worst 0.508 ULP over the test's 10 000 inputs (whole range,
  near 1, subnormals), against a 300-bit reference (log m =
  2·atanh((m−1)/(m+1))); `math.Log` measured 0.724 on the same inputs on
  arm64. `TestLogTable` checks each of the 128
  entries against the properties `log_data.c` documents (1/invc inside its
  subinterval, logc = log(c) within the rounding of invc, 0x1.8p9 + logc
  exact).
- **A second Go bug it removes**, known since 2022 as
  [golang/go#56600](https://github.com/golang/go/issues/56600) (fix pending in
  CL 448216): on amd64, `math.Log` is wrong for every subnormal input.
  `math.Log(5e-324)` returns −709.09 instead of −744.44, and `math.Log(1e-310)`
  −709.09 instead of −713.80 (NumPy, glibc: −744.44, −713.80). Its assembly
  inlines `Frexp` with bit masks that assume a normal input. Reproduced with Go
  1.26.8 and 1.27.1 on three x86 CPUs. `Log` and `Log10` inherited it; `TestLogSubnormal` pins the fix.
  (`Log2` already normalised through `math.Frexp` and was right.)

Speed, 16 Ki elements, one core: M4 37 µs (`math.Log` 72), Zen 3 73 µs
(`math.Log` 175, NumPy 69).

## ppc64le: VSX kernels (2026-10-04)

Go's ppc64 assembler has the VSX loads, stores and permutes but no vector
double arithmetic, so ppc64le ran the scalar code. go-asmgen v0.13.0 encodes
`xvadddp`, `xvsubdp`, `xvmuldp`, `xvdivdp`, `xvmaddadp`, `xvmaxdp`, `xvmindp`
and `xvsqrtdp` as `WORD`s, pinned against GNU as 2.44 and run bit-identical to
Go's arithmetic and `math.FMA` on POWER8 and POWER9. With them, go-ndarray has
VSX kernels for sum, dot, sqrt, add/sub/mul/div and the GEMM.

The GEMM micro-kernel (8×8, 32 accumulators) follows OpenBLAS's POWER9 dgemm:
no splats. Each A pair and its swap multiply each B column pair, so one
accumulator holds a diagonal (C[r][c], C[r+1][c+1]) and another the
anti-diagonal; the store rebuilds the rows with `xxpermdi`. On one POWER9 core,
measured in one binary: 13.3 GFLOP/s with `lxvdsx` splats, 14.8 with `xxpermdi`
splats, 15.4 with those unrolled by two, **16.9** with the diagonal scheme
unrolled by two (shipped).

POWER9 (cfarm29, 8 cores × SMT4), against the scalar code (best of 3–4
interleaved runs):

| op | before | after | gain |
|----|--:|--:|:--:|
| micro-kernel, 1 core | 8.3 GFLOP/s | 16.2 GFLOP/s | ×1.96 |
| serial GEMM 512², 1 core | 7.9 GFLOP/s | 15.8 GFLOP/s | ×2.0 |
| MatMul 1024², 32 threads | 39.2 ms | 23.6 ms | ×1.66 |
| MatMul 128² / 512² | | | ×1.8 / ×1.5 |
| MatVec 1024² | 823 µs | 267 µs | ×3.1 |
| Sum 1 Ki / 16 Ki / 256 Ki | | | ×6.6 / ×7.3 / ×3.2 |
| AddInto 1 Ki–256 Ki | | | ×2.7–4.1 |
| Sqrt, 1 core | | | ×1.05–1.18 |

At 4 Mi elements sum and add are memory-bound (×1.1). **Against OpenBLAS**
0.3.34 (its POWER9 kernel, loaded through `scipy-openblas64` on the same host;
the ppc64le NumPy wheel there has no BLAS): one thread 22.8 GFLOP/s at 512²,
so the serial GEMM is at 0.69×; 32 threads 13.1 ms at 1024², so MatMul is at
0.56×. OpenBLAS unrolls further and uses POWER9-only `lxv` loads; Go's ppc64le
baseline is POWER8.

## loong64: LASX kernels (2026-10-04)

Go's loong64 assembler has LASX vector float64 add/sub/mul/div, square root,
256-bit loads and the broadcast load (spelled as an arrangement,
`XVMOVQ off(R), X.V4`, which assembles to `xvldrepl.d`), but no fused
multiply-add; go-asmgen v0.14.0 encodes `xvfmadd.d` as a `WORD` (pinned
against GNU as 2.43, bit-identical to `math.FMA` on a Loongson 3C5000L;
transitional until cmd/asm names it). go-ndarray uses them for sum, dot, sqrt,
add/sub/mul/div and an 8×8 GEMM micro-kernel (16 four-lane accumulators, A
broadcast by `xvldrepl.d`). (v0.2.5 said Go had no broadcast load and emitted
it as a WORD too; it has one, and v0.2.6 emits the mnemonic, the same machine
code.) LASX is not on every LoongArch CPU, so the kernels run only when
the kernel reports it in AT_HWCAP (`/proc/self/auxv`), with the scalar code as
the fallback; the parsing is a pure function tested on every target.

Loongson 3C5000L (cfarm401, 32 cores), against the scalar code, best of 3
interleaved runs (the host's load average of ~146 is processes stuck in
uninterruptible sleep; its CPUs measured 100% idle):

| op | before | after | gain |
|----|--:|--:|:--:|
| micro-kernel, 1 core | 7.0 GFLOP/s | 27.5 GFLOP/s | ×3.95 |
| serial GEMM 512², 1 core | 5.3 GFLOP/s | 20.0 GFLOP/s | ×3.75 |
| MatMul 64² / 512² / 1024², 32 threads | | | ×1.9 / ×1.3 / ×1.1 |
| Sum 256 Ki / 4 Mi, 1 core | 658 / 11 750 µs | 65 / 4 615 µs | ×10.2 / ×2.5 |
| Sum 1 Ki / 16 Ki (below the parallel threshold: one core) | | | ×16 / ×13 |
| Sqrt 1 Ki–16 Ki | | | ×3.8 |
| MatVec 1024² | 364 µs | 224 µs | ×1.6 |

At 4 Mi elements in parallel, sum and sqrt are memory-bound (parity); the
parallel 4 Mi sum is bimodal on this host (1.6 ms or 7 ms in either version),
so only the minima are compared. No NumPy or BLAS is installable there (the
host has no route to package mirrors), so there is no external reference.

## Parallel GEMM: start once, wait on work (2026-10-04)

**Diagnosis, from hardware counters** (`perf stat`, Zen 3 guest cfarm421,
steady state, 1024², 3 s per configuration):

| workers | CPUs busy | IPC | instructions |
|---|---|---|---|
| 1 | 1.00 | 3.35 | 37.5 G |
| 2 | 1.91 | 2.00 | 37.6 G |
| 8 | 5.58 | 2.16 | 172 G (192 products) |
| 16 | 7.08 | 2.28 | 193 G (226 products) |

The instruction count does not grow, so nobody busy-waits. L1 misses do not
grow either. What changes is that at 16 workers the CPUs are busy only 44% of
the time. The driver forked and joined goroutines twice per (jc, pc) block of
the product, once to pack B and once for the row bands, 16 times in a 1024²
product. A fork/join wakes parked threads one after another. A micro-benchmark
on the same guest (16 goroutines, 700 µs of work each) measures its cost at
**26 µs with 2 workers, 165 µs with 8, 330 µs with 16**. 16 × 330 µs is ~40%
of a 12.5 ms product.

Two hypotheses were tested and **refuted** first:

- SMT siblings. The guest has 128 vCPUs, as many as the host has threads.
  Two serial GEMMs pinned on vCPU 16 and any other vCPU slow each other by
  0–10%, never by half.
- A block of A larger than L2. With MC = 256, the A block is 512 KiB, all of
  Zen 3's L2, against BLIS's MC = 72. Sweeping MC over {72, 120, 144, 192, 256}
  changed nothing beyond the noise at 2 workers, and gave +17% at 16 workers
  for 1024² but −8% for 512².

**The change.** The workers start once per product. They synchronise on
**work done**, never on workers arrived: a block's bands start once all of its
B panels are packed and all bands of the previous block are done, and the
packing is shared in claims of four NR-wide panels. A first version used a
barrier that waited for every worker, and it put the wake-up chain straight
back on the critical path for small products: 96³ on 8 POWER9 cores ran at
0.57×. B is double-buffered, so block i+1 is packed while block i is still
being multiplied. Each C tile is still accumulated in pc order, so the result
is bit-identical to the serial product. The pack buffers moved to two pools,
one for A and one for B: a pair per worker had held 16 MiB of unused B buffers
at 16 workers, which the collector empties and the next call zeroes again
(`memclr` page faults, 20% of the cycles in a KVM spinlock in one profile).

**Measured.** Every figure is the median of interleaved runs against the
previous driver.

| | 2 workers | 8 workers | 16 workers |
|---|---|---|---|
| Zen 3 guest, 1024² | 1.06× | 1.33× | 1.39× |
| Zen 3 guest, 512² | 1.20× | 1.20× | 1.33× |
| Zen 3 guest, 256² | 0.92× | 1.19× | 1.24× |
| Zen 3 guest, 128² | **0.89×** | 1.50× | 1.16× |
| Zen 3 guest, (1024×256)·(256×1024) | **0.82×** | 1.07× | 0.89× (p25 1.00×) |
| POWER9 bare metal, 512² | 1.34× | 1.37× | |
| POWER9 bare metal, 256² | 1.61× | 1.78× | |
| POWER9 bare metal, (1024×256)·(256×1024) | 1.13× | 1.18× | |

The Zen 3 runs were 12 rounds and the POWER9 runs 6, except 24 for 128² and
256² at 8 workers. Those two are bimodal on that machine, in both drivers, and
6 rounds had shown 256² at 0.72×. POWER9 used one thread per core (CPUs 0, 4,
8, …).

**Still behind.** On the Zen 3 guest, two workers lose 8–18% on mid-size
products, where a fork/join costs only 26 µs. Bare-metal POWER9 gains at two
workers, so this is not a property of the scheme, and no second code path was
added for it. Against OpenBLAS with 16 threads, measured three ways in one run:

| n | OpenBLAS-16 | previous driver | new driver |
|---|---|---|---|
| 256 | 708 µs | 1.06× | **1.44×** |
| 512 | 789 µs | 0.29× | 0.42× |
| 1024 | 4.60 ms | 0.43× | 0.51× |

(× = OpenBLAS time / ours, so higher is faster). On this guest, absolute times
drift through the day: the previous driver measured 0.54–0.66× of OpenBLAS in
the morning. Only ratios taken in the same run are comparable. Not measured on
Apple M-series, whose machine was shared with other jobs at the time. The bands
stay dynamic, so its slower efficiency cores are handled as before.

## SIMD coverage

- **amd64 (SSE2)** ships hand-vectorized `sum` (4-accumulator `ADDPD`), `sqrt`
  (packed `SQRTPD`), `max`/`min` (`MAXPD`/`MINPD` + `CMPPD` NaN scan), the
  **elementwise `add`/`sub`/`mul`/`div`** (packed `ADD/SUB/MUL/DIVPD`, 8
  doubles/iter + scalar tail), all at the GOAMD64=v1 baseline, and the **GEMM
  micro-kernel**: `gemmMicro6x8FMA` (6×8, 12 YMM accumulators, `VBROADCASTSD` +
  `VFMADD231PD`, C and A prefetched) when go-asmgen's CPUID probe reports FMA
  with the OS saving YMM state, else a pair of SSE2 6×4 `MULPD`+`ADDPD` tiles
  over the same packing. Generated by go-asmgen and validated per-arch in CI;
  the FMA kernel on real Cascade Lake and Zen 3 hosts.
- **arm64 (NEON)** ships hand-vectorized `sum`, `dot`, **packed `sqrt`**
  (`VFSQRT`), the **elementwise `add`/`sub`/`mul`/`div`** (`VFADD`/`VFSUB`/
  `VFMUL`/`VFDIV`), **`max`/`min`** (`VFMAX`/`VFMIN`, four D2 accumulators from
  8 elements up) and the **GEMM micro-kernel** (`gemmMicro4x8`: a 4×8 tile, 16 D2
  accumulators, using the **by-lane FMLA** `FMLA Vd.2D, Vn.2D, Vm.D[i]`). Only
  the indexed-element FMLA is still a raw `WORD` (16 of them): Go's assembler
  does not name it yet (pending as CL 764224).

  ⚠ **Corrected in v0.5.2.** Up to v0.5.1 this paragraph said Go's arm64
  assembler had no vector `FSQRT`, no plain vector FP add, and no vector form for
  `div`, so `add`/`sub`/`mul` went through an exact FMA against `1.0`, `sqrt` was
  a `WORD`, and `div` and `max`/`min` stayed scalar. Go 1.27 (the floor since
  v0.5.0) names all of them; the ISA always had them. Measured on one core,
  interleaved against v0.5.1 (8 rounds, median; `Sum`, `SqrtInto` and `AddInto`,
  unchanged, measured at 1.00× as the control):

  | one core | `DivInto` | `Max` | `MulInto` |
  |---|---|---|---|
  | Apple M (local) | 1.34–2.19× | 1.87–2.24× | 1.21× at 1 Ki, 1.00× above |
  | X-Gene (cfarm185) | 1.00× | 1.18–1.41× | 1.08–1.35× |

  X-Gene's vector divide is no faster than its scalar one. The sum, dot and
  GEMM C-accumulate still fold with an exact `VFMLA` against `1.0`, written
  before `VFADD` existed in Go; the result is the same.
- **ppc64le (VSX)**, since v0.2.3: sum, dot, sqrt, add/sub/mul/div and an 8×8
  GEMM micro-kernel; max/min stay scalar (the ISA's `xvmaxdp` NaN rule is not
  NumPy's). See the ppc64le section above.
- **loong64 (LASX)**, since v0.2.5, when the kernel reports LASX in AT_HWCAP
  (not every LoongArch CPU has it): sum, dot, sqrt, add/sub/mul/div and an 8×8
  GEMM micro-kernel; max/min stay scalar. See the loong64 section above.
- The other two 64-bit Go targets — **riscv64, s390x** — keep
  the validated scalar oracles, using the same four-accumulator max/min, direct
  sqrt loop, and a **scalar 4×4 GEMM micro-kernel** over the packed panels, and
  still get the **packing + cache blocking + multicore** structure (they have
  not been measured against NumPy). What the Go assembler offers them, checked
  on Go 1.26.4 and 1.27.1 by assembling and disassembling (2026-10-04):
  **ppc64le** has no vector-double arithmetic (no `XVADDDP`/`XVMADDADP`), which
  go-asmgen v0.13.0 now encodes as `WORD`s, so ppc64le has kernels;
  **loong64** has vector-double add/sub/mul/div (`VADDD` assembles to
  `vfadd.d`, `VMULD` to `vfmul.d`, `XVADDD` to `xvfadd.d`) and the broadcast
  load (`XVMOVQ off(R), X.V4` = `xvldrepl.d`), but no vector FMA, which
  go-asmgen v0.14.0 encodes (transitionally), so loong64 has kernels;
  **s390x** has them, FMA included (`VFADB`, `VFMADB`); **riscv64** has them
  (`VFADDVV`, `VFMACCVV`), but the V extension is optional and needs a run-time
  check. So s390x and riscv64 kernels are work not yet done, not a
  toolchain wall. (s390x additionally exercises the big-endian path in CI.)

All six are exercised in CI (native amd64/arm64 + qemu for the rest); each
per-arch job regenerates the committed `.s`, fails if it is stale, vets
(asmdecl), builds (cmd/asm encodes), and runs the bit/NaN-correctness suite. The
multicore path and the packed/cache-blocked GEMM driver are
architecture-independent; only the GEMM micro-kernel is per-arch (NEON 4×8 on
arm64, AVX2/FMA 6×8 or SSE2 on amd64, scalar 4×4 on the other four), each held to
the scalar ikj oracle in CI: exactly on integer-valued data, within tolerance on
general data, since above KC the block partials regroup the sum and the FMA
kernels round once per step where the scalar oracle may round twice.
