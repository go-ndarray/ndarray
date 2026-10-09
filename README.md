<p align="center"><img src="https://raw.githubusercontent.com/go-ndarray/brand/main/social/go-ndarray.png" alt="go-ndarray/ndarray" width="720"></p>

# ndarray — go-ndarray

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-013243)](https://go-ndarray.github.io/docs/)
[![Playground](https://img.shields.io/badge/playground-try%20it%20in%20your%20browser-013243)](https://go-ndarray.github.io/playground/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.27.2%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-numpy%20parity%20(float64)-9a6700)](docs/plan-ndarray.md)

**A pure-Go (CGO=0) NumPy-style N-dimensional array library.** Row-major
(C-order) strided arrays with:

- **Creation** — `New`/`Zeros`/`Ones`/`Full`/`FromData`/`Arange`/`Linspace`/
  `Eye`/`Identity`.
- **Shape & views** — `Reshape` (with `-1` inference), `Ravel`/`Flatten`,
  `Transpose`, `Squeeze`/`ExpandDims`, and **NumPy basic-indexing `Slice`**
  returning strided views that share data.
- **Elementwise** — `Add`/`Sub`/`Mul`/`Div` (+ scalar) with full **NumPy
  broadcasting**, `Map`/`Neg`/`Abs`, math **ufuncs** (`Sqrt`/`Exp`/`Log`/`Sin`/
  `Cos`/…), and broadcasting **comparisons** (`Greater`/`Equal`/… as 0/1 masks)
  plus `Maximum`/`Minimum`.
- **Reductions** — whole-array (`Sum`/`Prod`/`Max`/`Min`/`Mean`), per-axis
  (`SumAxis`/… with `keepdims`), index reductions (`ArgMax`/`ArgMin` flat and
  per-axis), cumulative scans (`CumSum`/`CumProd`), `Clip`, and `Where`.
- **Indexing** — basic-indexing `Slice` views plus **boolean/fancy indexing**:
  `MaskSelect` (`a[mask]`), `Nonzero` (`flatnonzero`), and `Take`.
- **Manipulation** — `Concatenate`/`Stack`/`VStack`/`HStack`.
- **Linear algebra** — `MatMul`/`Dot`/`Inner`/`Outer`.
- **Memory reuse** — `Workspace`, an arena for loops: bind the inputs with
  `ws.Use`, compute as usual, `ws.Reset()` at the end of each pass. Results stop
  costing the garbage collector anything (see below).

**Try it without installing anything:** the [playground](https://go-ndarray.github.io/playground/)
runs go-ndarray compiled to WebAssembly in your browser.

### Loops: `Workspace`

Every operation returns a new array, and in a loop those results are garbage
the collector must chase. NumPy frees its temporaries the moment their
reference count drops; Go cannot. On a Zen 3 that made `x + y` on 1 024
elements 3× slower than NumPy, and a chain like `sqrt(x*y + x)` 3.4× slower.
A `Workspace` gives the results of one pass a single arena that `Reset` hands
back:

```go
ws := ndarray.NewWorkspace()
for step := 0; step < steps; step++ {
	x := ws.Use(state)          // results computed from x come from ws
	p, _ := x.Mul(w)
	q, _ := p.Add(bias)
	state = q.Sqrt().Detach()   // the one result that outlives the pass
	ws.Reset()                  // everything else is recycled
}
```

After the first pass nothing is allocated, nothing is zeroed that the kernel
overwrites anyway, and the memory stays warm in cache: `x + y` on 1 024
elements goes from 2.6 µs to 0.38 µs (2.1× NumPy), `sqrt(x*y + x)` on 256 Ki
from 1.18 ms to 0.39 ms (7.5× NumPy). ⚠ An array allocated from the workspace
is invalid after `Reset`; `Detach` is how a result leaves it. The `*Into`
methods (`AddInto`, `SqrtInto`, …) remain the way to reuse one named buffer.

The numeric inner loops are kept behind a narrow kernel API. Behind it, large
elementwise ops, reductions and products run **multicore** (across
`GOMAXPROCS`). On **amd64, arm64, ppc64le, loong64, riscv64 and s390x** the hot
loops are [go-asmgen](https://github.com/go-asmgen)-generated SIMD kernels: sum,
add/sub/mul/div, sqrt and the dot product (SSE2 or AVX2/FMA on amd64, NEON on
arm64, VSX on ppc64le, LASX on loong64, RVV on riscv64 and the vector facility
on s390x, the last three when the CPU has them), max/min too on
amd64 and arm64, and a **panel-packed, cache-blocked GEMM** with an SIMD-FMA
micro-kernel (NEON 4×8; AVX2/FMA 6×8 chosen at run time, SSE2 fallback; VSX
8×8; LASX 8×8; RVV 6×8; s390x 4×8), the OpenBLAS/BLIS structure. The 32-bit
targets run the same pure-Go code those kernels are tested against. `Exp` and `Log` are ports of Arm's optimized-routines
(≤ 0.51 ULP measured, against up to 0.88 and 0.72 for Go's
`math.Exp`/`math.Log` on arm64),
and they are correct where amd64's `math.Exp` returns +Inf
([golang/go#81995](https://github.com/golang/go/issues/81995)) and `math.Log`
is wrong on subnormals
([golang/go#56600](https://github.com/golang/go/issues/56600)).

**Measured against NumPy 2.5.3 (OpenBLAS 0.3.34) on an AMD Zen 3, 16 cores**
(full tables in **[docs/perf.md](docs/perf.md)**):

- **Faster:** whole-array reductions (`Sum`/`Mean`/`Max` at 4 Mi: 2.3–4.5×),
  row reductions (`SumAxis(1)` 2.2×, `MaxAxis(1)` 1.7×), `Exp` from 256 Ki
  elements (2.4–5.6×), the `*Into` forms of `Add`/`Mul`/`Sqrt` (faster or at
  parity at every size), and `MatMul` against single-threaded OpenBLAS
  (3.7–5.9× from 512²); in a `Workspace`, `x + y` and `sqrt(x*y + x)` match or
  beat NumPy at every size measured.
- **At parity:** `SumAxis(0)`, `Inner` against 16-thread OpenBLAS, `Log` on
  one core.
- **Slower:** `MatMul` against 16-thread OpenBLAS at 512² and up (0.42× at
  512², 0.51× at 1024², measured in one run with OpenBLAS; at 256² ours is
  1.44× faster); `Dot` and mat·vec against NumPy's 16-thread BLAS (0.69× and 0.52× since v0.6.0's helper pool, measured in one run; 0.16× before); `Exp`
  below 256 Ki elements (0.82×); the *allocating* forms of elementwise ops up
  to 256 Ki elements, `Concatenate`/`Stack` and slice copies outside a
  `Workspace` (0.2–0.5×: Go's allocator and garbage collector, which
  `Workspace` largely removes).

On Apple silicon the GEMM reached parity with tuned BLAS at 1024² (OpenBLAS in
an arm64 VM, single-threaded vecLib on an M4 Max), and every product beats the
pure-Go `gonum` 4–10× (**[BENCHMARKS.md](BENCHMARKS.md)**). It is a
**standalone, reusable** module and the cgo-free ndarray backend behind
[go-embedded-ruby](https://github.com/go-embedded-ruby/ruby)'s `NDArray` class.

> ⚠️ **Status: float64 NumPy parity for the core surface.** Creation,
> slicing/views, broadcasting elementwise + ufuncs, reductions (incl.
> arg/cumulative/clip/where), manipulation, linear algebra and `Workspace` are
> complete, **100%-covered**, and differentially checked against NumPy. CI runs
> the suite on amd64, arm64, 386, and riscv64/loong64/ppc64le/s390x/arm under
> qemu, on Linux, macOS and Windows, and compiles it for every `GOOS/GOARCH`
> pair Go supports; v0.1.0 and v0.2.0 were also run on real amd64, arm64,
> ppc64le, riscv64 and loong64 hardware, v0.3.0 on real amd64, arm64, ppc64le
> and loong64. See **[docs/plan-ndarray.md](docs/plan-ndarray.md)**
> for the roadmap (more dtypes, more SIMD targets).

## Why this module?

[gonum](https://www.gonum.org/) is matrix-centric and its optimized assembly is
**amd64-only**. More broadly, **Ruby has no cgo-free ndarray** (`Numo::NArray`,
`NMatrix` are C extensions). A pure-Go core whose kernels are generated for every
arch is therefore a durable foundation. The numeric loops live in
`internal/kernels`, each with a pure-Go reference implementation;
[go-asmgen](https://github.com/go-asmgen)-generated SIMD kernels replace them
on every 64-bit target Go supports (amd64, arm64, ppc64le, loong64, riscv64,
s390x), behind the same API and held to the same tests.

## Goroutines

Operations on large arrays are spread over `GOMAXPROCS` goroutines. After the
first one, the package keeps up to `GOMAXPROCS-1` helper goroutines alive. When
an operation ends, they poll for the next one for 200 µs and then block, the way
OpenBLAS's threads do, so operations issued back to back do not pay to wake
threads. Waking threads per call made a 2^20-element dot slower on 8 POWER9 cores
than on 4. The helpers are never stopped: a goroutine-leak check (goleak) that
runs after ndarray operations will list them.

## Untrusted input

Shapes, indices and data can come from someone else (a file, a request), so
every operation answers a shape it cannot represent with an error wrapping
`ErrShapeMismatch` instead of a panic or a wrong array. That covers results as
well as inputs: `(2^32, 0) @ (0, 2^32)` holds no data, yet its result would
have 2^64 elements, so `MatMul` refuses it, as NumPy does. The one exception
is `Outer`, which has no error result: it panics when its result is too big,
which takes operands of more than 2^30 elements each. An empty array whose
other axis is huge, `(2^58, 0)`, costs nothing to reduce, concatenate or scan.

A shape that is representable can still be far larger than memory, and Go
cannot recover from running out of it: the process dies, where NumPy raises
`MemoryError`. Several operations produce more than they are given:
broadcasting (`(n, 1) + (1, n)` is n² elements), `Outer`, `MatMul`, and summing
the empty axis of `(0, n)` (n zeros, as in NumPy). When shapes come from
someone else, bound the result's size before computing it.

The assembly kernels take raw pointers. Their Go wrappers check every operand
the kernel reads or writes against the length it is given, so a mistake is an
index panic in Go, never an access outside a slice; a test places each operand
flush against an inaccessible page to prove the kernels themselves stay inside
(see [SECURITY.md](SECURITY.md)).

## Example

```go
import "github.com/go-ndarray/ndarray"

a, _ := ndarray.Arange(0, 6, 1)     // [0 1 2 3 4 5]
m, _ := a.Reshape(2, -1)            // -1 inferred -> [[0 1 2] [3 4 5]]

row, _ := ndarray.FromData([]float64{10, 20, 30}, 3)
sum, _ := m.Add(row)                // broadcast (2,3)+(3,) -> (2,3)

t := m.Transpose()                  // zero-copy (3,2) view
total := m.Sum()                    // 15
cols, _ := m.SumAxis(0, false)      // sum down each column -> [3 5 7]

// NumPy basic-indexing views (share data):
col0, _ := m.Slice(ndarray.All(), ndarray.A(0))     // m[:,0] -> [0 3]
sub, _ := m.Slice(ndarray.R(0, 2), ndarray.Step(2)) // m[0:2, ::2]
row1, _ := m.Slice(ndarray.A(1))                    // m[1]: unindexed axes are whole

// ufuncs and masks
roots := m.Sqrt()
two, _ := ndarray.Full(2, 1)
mask, _ := m.Greater(two)                           // broadcast 0/1 mask of m > 2

// linear algebra
b, _ := ndarray.Arange(0, 6, 1)
b, _ = b.Reshape(3, 2)
prod, _ := m.MatMul(b)              // (2,3)·(3,2) -> (2,2)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
