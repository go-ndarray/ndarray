# ndarray — go-ndarray

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-013243)](https://go-ndarray.github.io/docs/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![Go](https://img.shields.io/badge/go-1.26.4%2B-00ADD8)](https://go.dev/dl/)
[![Status](https://img.shields.io/badge/status-phase%200-9a6700)](docs/plan-ndarray.md)

**A pure-Go (CGO=0) NumPy-style N-dimensional array library.** Row-major
(C-order) strided arrays with constructors, reshape/transpose/copy, elementwise
operations with full **NumPy broadcasting**, mapping, and reductions — both
whole-array (`Sum`/`Prod`/`Max`/`Min`/`Mean`) and **along an axis** with
`keepdims` — with the numeric inner loops kept behind a narrow kernel API so
SIMD variants can drop in without changing callers.

It is a **standalone, reusable** module and the planned cgo-free ndarray backend
for [go-embedded-ruby](https://github.com/go-embedded-ruby/ruby).

> ⚠️ **Status: Phase 0.** The float64 core is complete and 100%-covered. See
> **[docs/plan-ndarray.md](docs/plan-ndarray.md)** for the architecture and the
> phased roadmap (dtypes, axis reductions, linalg/matmul, SIMD kernels via
> go-asmgen, Ruby binding).

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
m, _ := a.Reshape(2, 3)             // [[0 1 2] [3 4 5]]

row, _ := ndarray.FromData([]float64{10, 20, 30}, 3)
sum, _ := m.Add(row)                // broadcast (2,3)+(3,) -> (2,3)

t := m.Transpose()                  // zero-copy (3,2) view
total := m.Sum()                    // 15

cols, _ := m.SumAxis(0, false)      // sum down each column -> [3 5 7]
means, _ := m.MeanAxis(1, true)     // row means, keepdims -> shape (2,1)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
