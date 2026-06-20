<p align="center"><img src="https://raw.githubusercontent.com/go-ndarray/brand/main/social/go-ndarray.png" alt="go-ndarray/ndarray" width="720"></p>

# ndarray — go-ndarray

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-013243)](https://go-ndarray.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
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
- **Reductions** — whole-array (`Sum`/`Prod`/`Max`/`Min`/`Mean`) and per-axis
  (`SumAxis`/… with `keepdims`).
- **Manipulation** — `Concatenate`/`Stack`/`VStack`/`HStack`.
- **Linear algebra** — `MatMul`/`Dot`/`Inner`/`Outer`.

The numeric inner loops are kept behind a narrow kernel API so SIMD variants can
drop in without changing callers. It is a **standalone, reusable** module and
the planned cgo-free ndarray backend for
[go-embedded-ruby](https://github.com/go-embedded-ruby/ruby).

> ⚠️ **Status: float64 NumPy parity for the core surface.** Creation, slicing/
> views, broadcasting elementwise + ufuncs, reductions, manipulation, and
> scalar-kernel linear algebra are complete, **100%-covered**, and
> differentially checked against numpy. See
> **[docs/plan-ndarray.md](docs/plan-ndarray.md)** for the roadmap (more dtypes,
> SIMD kernels via go-asmgen, Ruby binding).

## Why this module?

[gonum](https://www.gonum.org/) is matrix-centric and its optimized assembly is
**amd64-only**. More broadly, **Ruby has no cgo-free ndarray** (`Numo::NArray`,
`NMatrix` are C extensions). A pure-Go core whose kernels are generated for every
arch is therefore a durable foundation. The numeric loops live in
`internal/kernels`; **Phase 1** replaces them with
[go-asmgen](https://github.com/go-asmgen)-generated SIMD kernels across all six
64-bit Go targets (amd64, arm64, riscv64, loong64, ppc64le, s390x), selected at
runtime, behind the same API and the same tests.

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
