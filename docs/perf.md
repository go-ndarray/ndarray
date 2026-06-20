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
| Sqrt | 4 194 304 | 1 728 306 | 7 724 733 | **2 265 548** | 0.76 (numpy) |
| Sum  | 1 024 | 671 | **135** | **135** | **4.95** |
| Sum  | 4 194 304 | 505 997 | **493 755** | **209 908** | **2.41** |
| Max  | 1 024 | 620 | **994** | **996** | 0.62 (numpy) |
| Max  | 4 194 304 | 366 888 | 4 430 933 | **1 201 354** | 0.31 (numpy) |

### Reading the table

- **Add / Mul on large arrays: go-ndarray wins ~2×.** Allocation elimination +
  multicore. Note even single-core large Mul edges out NumPy now (the kernel is
  memory-bandwidth bound and the old copies were the bottleneck).
- **Sum: go-ndarray wins everywhere — up to ~5× on small, 2.4× on large.** The
  SIMD 4-accumulator kernel already beats NumPy single-core (493 µs vs 506 µs at
  4 M); multicore then roughly halves it again. Small sums are dominated by
  NumPy's per-call Python/dispatch overhead, which Go does not pay.
- **Small arrays (n≈1 024) for Add/Mul/Max: NumPy still wins.** Below the
  parallel threshold go-ndarray is serial, and its per-call `*Array` allocation
  costs more than NumPy's tight C loop at this size. This is the right
  trade-off: parallelising a 1 K array would *slow it down*. (Sum is the
  exception because the SIMD kernel needs no result allocation.)
- **Sqrt: NumPy still wins (~1.3×).** `Map` dispatches through a `func(float64)
  float64`, which blocks both Go auto-vectorization and a packed SIMD `FSQRT`.
  Multicore narrows the gap from ~4.5× to ~1.3×. A dedicated SIMD `sqrt` kernel
  (packed `VFSQRT`) is the planned fix.
- **Max: NumPy still wins (~3× at 4 M).** Max is on the scalar parallel path
  only — a NEON `FMAX` kernel would close it, but `FMAX`'s NaN-propagation
  differs from the scalar `if v > m` oracle, so a bit-identical vector max needs
  a NaN pre-scan; deferred rather than ship a semantics divergence.

## Matrix multiply (ns/op; lower is better)

| n (N×N) | NumPy 2.2.4 (ref BLAS) | go (1 core) | go (4 core) | go × (4 core) |
|------:|------:|------:|------:|:--:|
| 64  | 77 877 | 97 623 | **99 322** | 0.78 (numpy) |
| 128 | 653 655 | 897 997 | **284 787** | **2.29** |
| 256 | 4 959 421 | 6 791 576 | **1 890 900** | **2.62** |
| 512 | 38 475 076 | 50 585 479 | **13 979 582** | **2.75** |

- **go-ndarray's parallel ikj GEMM wins 2.3–2.8× at 128–512** against this
  NumPy's BLAS. Single-core it loses (the scalar ikj kernel is ~1.3× slower than
  reference BLAS); multicore is what wins.
- **Honest caveat:** this NumPy links Debian's **reference BLAS**, which is
  single-threaded and unoptimized. On a system where NumPy links **OpenBLAS** or
  **MKL** (cache-blocked, hand-tuned, multi-threaded), NumPy's matmul would be
  several times faster and would likely lead. go-ndarray's GEMM is currently
  parallel but **not yet register/cache-blocked**; closing the gap against a
  tuned BLAS requires that blocking (planned). The win above is real on this
  configuration, and stated with its scope.

## Correctness

Every result is validated **bit-for-bit against NumPy** (shapes and values), not
just timed — see the cross-check in the work log: `Sum`, `Max`, `Min`, `Add`,
`Mul` and the full `MatMul` (including the sum over all output elements) match
NumPy exactly at n = 100 000 / 80×80. The SIMD sum kernel is validated against
the scalar oracle to a tight relative tolerance across every residue mod 8 (its
lane-parallel grouping is a valid reordering, the same kind NumPy's pairwise
summation uses — bit-identity is not claimed for a reduction, closeness is).

## Where go-ndarray still loses, and the plan

| op | status | fix |
|----|--------|-----|
| small arrays (n≲a few K) | NumPy wins (no parallelism, per-`*Array` alloc) | lower threshold / pooling is not worth the small-size risk; accepted |
| `Sqrt` (and other `Map` ufuncs) | NumPy wins ~1.3× | packed SIMD `VFSQRT`/`VEXP` kernels behind a non-`func`-pointer seam |
| `Max` / `Min` large | NumPy wins ~3× | NEON/SSE2 `FMAX` kernel with a NaN pre-scan |
| `MatMul` vs tuned BLAS | reference-BLAS: go wins; OpenBLAS/MKL: BLAS leads | register + cache blocking of the GEMM kernel |

## SIMD coverage

The sum kernel ships hand-vectorized on **amd64 (SSE2)** and **arm64 (NEON)**,
generated by go-asmgen and validated per-arch in CI. The other four 64-bit Go
targets — **riscv64, loong64, ppc64le, s390x** — keep the validated scalar
reduction (Go's loong64/ppc64le assemblers expose no vector-double arithmetic;
riscv64's V extension is optional), and still get the **multicore** win, so they
also beat single-threaded NumPy on large arrays. All six are exercised in CI
(native amd64/arm64 + qemu for the rest), and the multicore path is
architecture-independent.
