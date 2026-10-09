# Design — dtypes

> Status: accepted, implementation in progress. Supersedes decision 4 of
> [plan-ndarray.md](plan-ndarray.md) ("`float64` first").

## Why now

`Array` holds `float64` only. Three things the library is meant to do cannot be
done on that alone:

- **Spectra.** `go-fft` returns `[]complex128`. Without a complex dtype an FFT
  result cannot be an array, so the two libraries do not compose.
- **Masks.** Comparisons return `0/1` floats. They work for `MaskSelect` and
  `Where`, but a mask cannot be told apart from data, and it costs 8 bytes per
  element instead of 1.
- **Memory-bound work.** `float32` halves the bytes moved, which is the whole
  cost of the elementwise ops; images arrive as `uint8`; counts and indices are
  integers.

The type at the centre of the library is the one thing that gets more
expensive to change every release, and the module is pre-1.0. That is why it
goes first.

## Decision: the dtype is a runtime property of the one `Array` type

```go
type DType uint8 // Bool, Int8…Int64, Uint8…Uint64, Float32, Float64, Complex64, Complex128

a, _ := ndarray.Zeros(3, 4)          // Float64, as before
b := a.AsType(ndarray.Complex128)
c, _ := a.Add(b)                     // Complex128, by NumPy promotion
m, _ := a.Greater(ndarray.Scalar(0)) // Bool
```

The alternative was a generic `Dense[T]`. It was rejected for four reasons:

1. **Promotion cannot be written as a generic method.** `int32 / int32` is
   `float64` and `int8 + float32` is `float32`: the result type depends on both
   operands. A method on `Dense[T]` cannot return `Dense[U]` for a `U` chosen
   from its argument, and methods cannot have their own type parameters. The
   arithmetic would have to move to package functions, one per pair of types
   in the general case, and NumPy's semantics would be lost or re-encoded.
2. **The consumers are dynamic.** The notebook evaluates cells in an
   interpreter, and the Ruby binding (`go-embedded-ruby`) is a dynamic language.
   Both hold "an array" and ask it what it is; neither can carry a type
   parameter.
3. **Nothing that exists breaks.** `*Array`, its methods and the `float64` hot
   path keep their names, signatures and speed. The Ruby binding uses method
   expressions such as `(*nd.Array).Sqrt`; those would stop compiling on an
   alias of an instantiated generic type.
4. **Static typing is still available where it pays**: generic accessors
   `Data[T]`, `FromSlice[T]` and `Item[T]` check the dtype once at the boundary
   and hand back a `[]T`.

## Storage

```go
type Array struct {
	data    []float64 // Float64 storage — the existing hot path, unchanged
	ext     any       // []T for every other dtype, nil for Float64
	dtype   DType
	shape   []int
	strides []int
	offset  int
}
```

Keeping `data []float64` as a concrete field means every existing `float64`
kernel, SIMD path and benchmark is untouched. The other dtypes are reached
through one type assertion per operation, not per element. Shape, strides and
views do not depend on the dtype, so slicing, reshape, transpose, broadcast and
the manipulation functions are written once over a generic helper.

Inner loops for the other dtypes are generic Go (`func add[T number](dst, x,
y []T)`). The gc compiler stencils one copy per GC shape, and every numeric type
has its own shape, so these are fully specialised loops. SIMD kernels come
dtype by dtype after correctness, `float32` first.

## Semantics: NumPy 2.x, checked against NumPy

- **Array–array promotion** follows `numpy.result_type`. The full 13×13 table
  is generated from NumPy 2.5.3 and committed as a test fixture
  (`testdata/promotion.json`, embedded so the emulated CI arches can read it);
  the implementation is tested against every cell.
- **Go scalars are weak**, like Python scalars under NEP 50: `AddScalar(1.5)` on
  a `float32` array stays `float32`, on an `int8` array becomes `float64`.
  `ScalarOf(v, dtype)` makes a strong one.
- **True division** of integers or bools gives `float64`.
- **Reductions:** `Sum`/`Prod`/`CumSum` of bool and signed ints accumulate in
  `int64`, of unsigned ints in `uint64`; `Mean` of integers is `float64`;
  `Max`/`Min` keep the dtype; arg-reductions are `int64`.
- **Float ufuncs on integers** (`Sqrt`, `Exp`, `Sin` …) give `float64` for
  32- and 64-bit ints, `float32` for 16-bit ints.
- **Comparisons return `Bool`.** `MaskSelect` and `Where` accept a `Bool` mask
  and, as NumPy does, any other dtype by truthiness, so existing `0/1` masks
  keep working.
- **Integer overflow wraps**, as it does in both Go and NumPy.

### Deliberate differences

| NumPy | here | why |
|---|---|---|
| `float16` exists; `sqrt(int8)` is `float16` | no `float16`; `sqrt(int8)`, `sqrt(uint8)` and `sqrt(bool)` give `float32` | Go has no half-precision arithmetic; `float32` is the smallest type that holds the result exactly |
| `round` is half-to-even | half-away-from-zero (unchanged) | documented since v0.1.0 |
| `datetime64`, `str_`, `object`, structured dtypes | absent | out of scope for a numeric library |

## Element access

- `At(idx...) float64` keeps working for every real dtype, converting the
  element. On a complex array it panics with a message that names `AtComplex`.
  NumPy instead discards the imaginary part with a `ComplexWarning`; Go has no
  warnings, and a silent loss is the defect that warning exists to flag.
- `AtComplex(idx...) complex128`, and `Item[T](a, idx...) T` for exact typed
  access.
- `Set(v float64, idx...)` converts into the array's dtype.

## Breaking changes

| before | after | migration |
|---|---|---|
| comparisons return a `Float64` 0/1 array | they return `Bool` | `.AsType(Float64)` to get the old values; `MaskSelect` and `Where` need no change |
| `MaskSelect` takes a 0/1 float mask | it takes any dtype (truthiness) | none |

The only consumer in the fleet, `go-embedded-ruby/ruby` (`internal/vm/ndarray.go`),
does not use comparisons; it is pinned to a June pseudo-version and will be
moved forward in its own pull request.

## Delivery

1. **Core** — `DType`, storage, `AsType`, creation with a dtype
   (`ZerosOf`/`OnesOf`/`FullOf`/`FromSlice[T]`), element access, and every
   shape-only operation (reshape, transpose, slice, copy, flatten, expand,
   squeeze, concatenate, stack, string form) for every dtype.
2. **Arithmetic** — promotion, the four operators and the scalar forms,
   comparisons returning `Bool`, `Where`/`MaskSelect`, reductions with their
   accumulator types, unary ufuncs.
3. **Complex** — `Real`/`Imag`/`Conj`/`Angle`, `Abs` of complex to float, and an
   `fft` subpackage bridging `go-fft` on arrays (`fft.FFTN(a, axes...)`).
4. **Speed** — `float32` SIMD kernels, then `MatMul` for `float32` and complex.

Each step keeps the 100% coverage gate and the per-arch CI, and is checked
against NumPy before it ships.
