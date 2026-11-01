# Coverage against NumPy, SciPy and MATLAB

Inventory of what `go-ndarray/ndarray` (v0.31.0, with its `linalg`, `random`
and `fft` subpackages) implements against the public API of NumPy 2.5.3
(`numpy`, `numpy.linalg`, `numpy.random`) and `scipy.linalg` v1.18.1, with
MATLAB's core matrix functions as a third column. Status: **done** (same
semantics and the options that matter), **partial** (what is missing is
named), **missing**. "Dep" is what a missing item needs beyond this package.
`numpy.fft` and `scipy.fft` are inventoried in `go-fft/fft`
(`docs/coverage.md`); this package's `fft` subpackage is a bridge to it and is
listed under "Bridges" below.

Sources. NumPy: `numpy/__init__.py` at tag v2.5.3 (the `from ._core import`
block and the `lib` import blocks, line by line), `numpy/linalg/__init__.py`
and `numpy/random/__init__.py` at the same tag. SciPy: the autosummary of
`scipy/linalg/__init__.py` at tag v1.18.1. Go side: `go doc -all` of
`ndarray`, `ndarray/linalg`, `ndarray/random`, `ndarray/fft`, `README.md`,
`docs/plan-ndarray.md`, `docs/linalg.md`, `docs/random.md` and the sources on
main. MATLAB: the MathWorks function lists are rendered client-side and could
not be read, so the MATLAB column comes from knowledge of the documented core
function names and must be re-checked against the MathWorks reference before a
slice is started.

Conventions that explain many rows. `Array` is one runtime-typed type over 13
dtypes (bool, int8..64, uint8..64, float32/64, complex64/128), row-major with
strided views; there is no Fortran order, no float16, no longdouble, no
object, string or datetime dtype. Most operations are methods of `*Array`
(`a.Add(b)`), not package functions; "Go" names the method. Many reductions
come in a whole-array form returning a scalar (`Sum`) and an axis form
(`SumAxis`), the latter taking `keepdims`. Errors are returned, not raised.

## Totals

Computed by a throwaway Go program from the tables of this file: one count per
backticked name in the first column. See "Counting" at the end.

| Section | done | partial | missing | names |
|---|---|---|---|---|
| numpy top level (the 467 names of the import blocks of `numpy/__init__.py`, plus 14 aliases and extras) | 84 | 76 | 321 | 481 |
| numpy.linalg | 18 | 7 | 6 | 31 |
| numpy.random, Generator | 42 | 0 | 3 | 45 |
| numpy.random, bit generators and seeding | 10 | 0 | 0 | 10 |
| numpy.random, legacy module-level functions | 12 | 12 | 26 | 50 |
| scipy.linalg | 7 | 13 | 85 | 105 |
| MATLAB core (names not already listed above) | 16 | 0 | 32 | 48 |

## numpy: creation

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `zeros`, `ones`, `full` | `zeros`, `ones` | `Zeros`, `ZerosOf`, `Ones`, `OnesOf`, `Full`, `FullOf` | done | |
| `empty` | `zeros` | `New` (zero-filled) | done | |
| `array`, `asarray` | `[ ]` | `FromData`, `FromSlice`, `Scalar` | partial | nested or jagged input, `dtype=`, `copy=`, `ndmin=` |
| `asanyarray`, `ascontiguousarray`, `asfortranarray`, `require` | | | missing | Fortran order is not supported (design: row-major only) |
| `empty_like`, `zeros_like`, `ones_like`, `full_like` | `zeros(size(A))` | | missing | none (shape and dtype of an existing array) |
| `arange` | `a:s:b` | `Arange` | partial | float64 only: no `dtype`, no integer or one-argument forms |
| `linspace` | `linspace` | `Linspace` | partial | `endpoint`, `retstep`, `axis`, array `start`/`stop`, `dtype` |
| `logspace`, `geomspace` | `logspace` | | missing | `Pow`/`Log` ufuncs (have `Exp`, `Log`, `Power`) |
| `eye`, `identity` | `eye` | `Eye`, `Identity` | done | `dtype` form: `OnesOf` plus fill |
| `diag`, `diagflat`, `tri`, `tril`, `triu`, `vander` | `diag`, `tril`, `triu`, `vander` | | missing | none |
| `indices`, `fromfunction`, `fromiter` | `ndgrid` | | missing | none |
| `frombuffer`, `fromfile`, `fromstring`, `fromregex` | `fread`, `textscan` | `FromSlice`, `UnmarshalBinary` | partial | raw buffers with dtype and byte order; text parsing |
| `copy` | | `Copy` | done | |
| `copyto` | | `AddInto` family only | missing | none (general `copyto` with `where`) |
| `astype` | `cast` | `AsType` | done | |
| `meshgrid` | `meshgrid`, `ndgrid` | | missing | none |
| `mgrid`, `ogrid`, `r_`, `c_`, `s_`, `index_exp`, `ix_` | `[ ]`, `:` | `Slice` with `A`, `R`, `Rng` | partial | `mgrid`, `ogrid`, `r_`, `c_`, `ix_` missing; `s_` / `index_exp` are `Index` builders |
| `from_dlpack` | | | missing | none (zero-copy exchange) |

## numpy: shape and joining

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `reshape` | `reshape` | `Reshape` | done | `order='F'` |
| `ravel` | `A(:)` | `Ravel` | partial | returns a copy where NumPy returns a view when it can |
| `squeeze` | `squeeze` | `Squeeze` | done | |
| `expand_dims` | | `ExpandDims` | done | |
| `transpose` | `permute`, `'` | `Transpose` | partial | reverses the axes only; no `axes` permutation |
| `permute_dims`, `matrix_transpose`, `swapaxes`, `moveaxis`, `rollaxis` | `permute` | | missing | none (a strided view with permuted shape and strides) |
| `concatenate`, `concat` | `cat` | `Concatenate` | done | |
| `stack` | `cat(3, ..)` | `Stack` | done | |
| `hstack` | `horzcat` | `HStack` | done | |
| `vstack` | `vertcat` | `VStack` | done | |
| `dstack`, `column_stack`, `block`, `unstack` | `cat` | | missing | none |
| `split`, `array_split`, `hsplit`, `vsplit`, `dsplit` | | | missing | none (views) |
| `tile`, `repeat`, `resize` | `repmat`, `repelem` | | missing | none |
| `pad` | `padarray` (Image Processing Toolbox) | | missing | none |
| `delete`, `insert`, `append`, `trim_zeros` | `A(idx)=[]` | | missing | none |
| `flip`, `fliplr`, `flipud`, `rot90`, `roll` | `flip`, `fliplr`, `flipud`, `rot90`, `circshift` | `Slice` with negative `Step` for flips | partial | named functions and `roll`, `rot90` missing |
| `atleast_1d`, `atleast_2d`, `atleast_3d` | | | missing | none |
| `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`, `broadcast` | `repmat`, implicit expansion | broadcasting inside binary ops | missing | none (public view with zero strides) |
| `shape`, `ndim`, `size` | `size`, `ndims`, `numel` | `Shape`, `Ndim`, `Size` | done | |
| `apply_along_axis`, `apply_over_axes` | `cellfun`, `arrayfun` | | missing | none |
| `kron` | `kron` | | missing | none |

## numpy: elementwise math (ufuncs)

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `add`, `subtract`, `multiply` | `+`, `-`, `.*` | `Add`, `Sub`, `Mul`, `AddScalar`, `SubScalar`, `MulScalar`, `AddInto` | done | `out=`, `where=`, `casting=` keywords (only `*Into` forms) |
| `divide`, `true_divide` | `./` | `Div`, `DivScalar`, `DivInto` | done | |
| `negative` | `-` | `Neg` | done | |
| `absolute`, `abs` | `abs` | `Abs` | done | |
| `sqrt` | `sqrt` | `Sqrt`, `SqrtInto` | done | |
| `square` | `.^2` | `Square` | done | |
| `exp` | `exp` | `Exp` | done | float32 runs the float64 kernel |
| `log` | `log` | `Log` | done | |
| `log2` | `log2` | `Log2` | done | within 2 ULP of NumPy |
| `log10` | `log10` | `Log10` | done | |
| `sin`, `cos`, `tan` | `sin`, `cos`, `tan` | `Sin`, `Cos`, `Tan` | done | |
| `ceil`, `floor` | `ceil`, `floor` | `Ceil`, `Floor` | done | |
| `conjugate`, `conj` | `conj` | `Conj` | done | |
| `maximum`, `minimum` | `max`, `min` | `Maximum`, `Minimum` | done | |
| `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` | `==`, `~=`, `>`, `>=`, `<`, `<=` | `Equal`, `NotEqual`, `Greater`, `GreaterEqual`, `Less`, `LessEqual` | done | |
| `logical_and`, `logical_or`, `logical_xor`, `logical_not` | `&`, `\|`, `xor`, `~` | `LogicalAnd`, `LogicalOr`, `LogicalXor`, `LogicalNot` | done | |
| `power`, `pow` | `.^` | `Power` | partial | scalar exponent only: no array exponent, no integer-power semantics |
| `around`, `round` | `round` | `Round` | partial | rounds half away from zero (NumPy: half to even); no `decimals` |
| `clip` | `min(max(..))`, `rescale` | `Clip` | partial | array bounds, one-sided (`a_min` or `a_max` alone) |
| `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `asinh`, `acosh`, `atanh` | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | | missing | scalar kernels (go-numerics/special or math), SIMD later |
| `arcsin`, `arccos`, `arctan`, `asin`, `acos`, `atan` | `asin`, `acos`, `atan` | | missing | same |
| `arctan2`, `atan2` | `atan2` | | missing | same |
| `exp2`, `expm1`, `log1p`, `logaddexp`, `logaddexp2`, `cbrt` | `expm1`, `log1p`, `cbrt` | | missing | same (`math.Expm1`, `Log1p`, `Cbrt` exist in Go) |
| `hypot` | `hypot` | | missing | `math.Hypot` |
| `deg2rad`, `rad2deg`, `degrees`, `radians` | `deg2rad`, `rad2deg` | | missing | none |
| `reciprocal`, `positive`, `sign`, `signbit`, `fabs`, `copysign`, `heaviside`, `spacing`, `nextafter` | `sign` | | missing | none |
| `floor_divide`, `mod`, `remainder`, `fmod`, `divmod`, `float_power` | `floor`, `mod`, `rem` | | missing | none (integer semantics per dtype) |
| `fmax`, `fmin` | `max`, `min` (omitnan) | | missing | none |
| `frexp`, `ldexp`, `modf`, `rint`, `trunc`, `fix` | `fix` | | missing | none |
| `gcd`, `lcm` | `gcd`, `lcm` | | missing | none (integer dtypes) |
| `isfinite`, `isinf`, `isnan`, `isposinf`, `isneginf`, `isnat` | `isfinite`, `isinf`, `isnan` | | missing | none |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_invert`, `invert`, `left_shift`, `right_shift`, `bitwise_left_shift`, `bitwise_right_shift`, `bitwise_count` | `bitand`, `bitor`, `bitxor`, `bitshift` | | missing | none (integer dtypes) |
| `angle` | `angle` | `Angle` | done | `deg=True` |
| `real`, `imag` | `real`, `imag` | `Real`, `Imag` | done | |
| `sinc`, `i0`, `unwrap` | `sinc`, `besseli`, `unwrap` | | missing | Bessel I0 (go-numerics/special) |
| `kaiser`, `bartlett`, `blackman`, `hamming`, `hanning` | `kaiser`, `bartlett`, `blackman`, `hamming`, `hann` | `fft.Bartlett`, `Blackman`, `Hamming`, `Hann` in go-fft | partial | not exposed from this package; `kaiser` needs `i0` |
| `isclose`, `allclose`, `array_equal`, `array_equiv` | `isequal`, `ismembertol` | | missing | none |
| `nan_to_num`, `real_if_close`, `asarray_chkfinite` | | | missing | none |
| `iscomplex`, `isreal`, `iscomplexobj`, `isrealobj`, `isscalar`, `iterable` | `iscomplex`, `isreal`, `isscalar` | `DType.IsComplex` | partial | the object-level predicates |
| `vectorize`, `frompyfunc` | `arrayfun` | `Map` | partial | `Map` is float64 to float64 on one array; no multi-input, no dtype list |
| `ufunc` | | | missing | no user-defined ufunc framework (`reduce`, `accumulate`, `outer`, `at`) |
| `piecewise`, `select`, `choose` | | | missing | none |

## numpy: reductions and statistics

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `sum` | `sum` | `Sum`, `SumAll`, `SumAxis` | partial | multiple axes (tuple), `initial`, `where`, `dtype`; `SumAxis` takes one axis |
| `prod` | `prod` | `Prod`, `ProdAll`, `ProdAxis` | partial | same |
| `mean` | `mean` | `Mean`, `MeanAxis` | partial | same |
| `max`, `amax` | `max` | `Max`, `MaxAxis` | partial | multiple axes, `initial`, `where` |
| `min`, `amin` | `min` | `Min`, `MinAxis` | partial | same |
| `argmax`, `argmin` | `max`, `min` (index output) | `ArgMax`, `ArgMaxAxis`, `ArgMin`, `ArgMinAxis` | done | |
| `cumsum`, `cumprod` | `cumsum`, `cumprod` | `CumSum`, `CumSumFlat`, `CumProd`, `CumProdFlat` | done | `dtype` |
| `cumulative_sum`, `cumulative_prod` | | `CumSum`, `CumProd` | partial | `include_initial` |
| `all`, `any` | `all`, `any` | `All`, `Any` | partial | `axis`, `keepdims`, `where` (whole-array only) |
| `count_nonzero` | `nnz` | | missing | none |
| `ptp` | `range`, `peak2peak` | | missing | none |
| `std`, `var` | `std`, `var` | | missing | none (two-pass or Welford; `ddof`) |
| `median`, `average`, `percentile`, `quantile` | `median`, `prctile`, `quantile` | | missing | partition/sort |
| `nanargmax`, `nanargmin`, `nancumprod`, `nancumsum`, `nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar` | `max`, `sum`, `mean`... with `'omitnan'` | | missing | none |
| `cov`, `corrcoef` | `cov`, `corrcoef` | | missing | none |
| `histogram`, `histogram2d`, `histogramdd`, `histogram_bin_edges`, `bincount`, `digitize` | `histcounts`, `histcounts2`, `discretize` | | missing | `searchsorted` |
| `diff`, `ediff1d`, `gradient` | `diff`, `gradient` | | missing | none |
| `trapezoid` | `trapz` | | missing | none |
| `interp` | `interp1` | | missing | `searchsorted` |
| `convolve`, `correlate` | `conv`, `xcorr` | | missing | none (direct); FFT path via go-fft |

## numpy: sorting, searching, sets

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `sort`, `argsort`, `lexsort`, `sort_complex`, `partition`, `argpartition` | `sort`, `sortrows`, `maxk`, `mink` | | missing | sort kernels (pdqsort / radix), stable sort; NaN ordering |
| `searchsorted` | `discretize`, `interp1` | | missing | none |
| `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` | `unique` | | missing | sort |
| `intersect1d`, `setdiff1d`, `setxor1d`, `union1d`, `isin` | `intersect`, `setdiff`, `setxor`, `union`, `ismember` | | missing | sort |
| `nonzero` | `find` | `Nonzero` | partial | returns flat indices (`flatnonzero`), not a tuple of per-axis indices |
| `flatnonzero` | `find` | `Nonzero` | done | |
| `argwhere` | `find` with `ind2sub` | | missing | none |
| `where` | `merge`, logical indexing | `Where` | partial | one-argument form (`where(cond)`) |
| `extract`, `compress`, `place`, `put`, `putmask` | `A(mask)` | `MaskSelect` | partial | `extract` is `MaskSelect`; others missing |
| `take` | `A(idx)` | `Take` | partial | flat indices only: no `axis`, `out`, `mode` |
| `take_along_axis`, `put_along_axis` | | | missing | none |
| `diag_indices`, `diag_indices_from`, `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from`, `mask_indices`, `fill_diagonal` | `tril`, `triu` | | missing | none |
| `ravel_multi_index`, `unravel_index` | `sub2ind`, `ind2sub` | | missing | none |
| `ndenumerate`, `ndindex`, `nditer`, `nested_iters`, `flatiter` | | | missing | iterators (Go range-over-func) |

## numpy: linear algebra in the top-level namespace

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `dot` | `dot`, `*` | `Dot` | partial | 1-D and 2-D only; N-D `dot` (sum over last axis of a and second-to-last of b) |
| `matmul` | `*`, `pagemtimes` | `MatMul` | partial | 2-D only: no stacked (batched) matmul, no 1-D operands, no `@` broadcasting |
| `inner` | | `Inner` | partial | 1-D and 2-D only |
| `outer` | | `Outer` | done | |
| `vdot`, `vecdot`, `matvec`, `vecmat` | `dot` | | missing | none |
| `tensordot`, `einsum`, `einsum_path` | `tensorprod` | | missing | contraction planner, GEMM reuse |
| `cross` | `cross` | | missing | none |
| `trace`, `diagonal` | `trace`, `diag` | | missing | none (view for `diagonal`) |

## numpy: dtypes and type information

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `dtype` | `class` | `DType` | partial | structured and flexible dtypes |
| `ndarray` | | `Array` | partial | see rows above |
| `bool`, `bool_`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float32`, `float64`, `complex64`, `complex128` | `logical`, `int8`, ..., `single`, `double` | `Bool`, `Int8`..`Int64`, `Uint8`..`Uint64`, `Float32`, `Float64`, `Complex64`, `Complex128` | done | |
| `byte`, `ubyte`, `short`, `ushort`, `intc`, `uintc`, `int_`, `uint`, `intp`, `uintp`, `long`, `ulong`, `longlong`, `ulonglong`, `single`, `double`, `csingle`, `cdouble` | | | missing | platform alias names for `ParseDType` |
| `half`, `float16` | `half` | | missing | a float16 type and conversions (no Go native); SIMD via F16C / NEON |
| `longdouble`, `clongdouble`, `float96`, `float128`, `complex192`, `complex256` | | | missing | extended precision (double-double or go-bignum) |
| `str_`, `bytes_`, `object_`, `void`, `record`, `recarray`, `character`, `flexible` | `string`, `char`, `struct` | | missing | non-numeric dtypes: a different storage model |
| `datetime64`, `timedelta64`, `datetime_as_string`, `datetime_data`, `busday_count`, `busday_offset`, `busdaycalendar`, `is_busday` | `datetime`, `duration`, `busdays` | | missing | datetime dtypes |
| `generic`, `number`, `integer`, `inexact`, `floating`, `complexfloating`, `signedinteger`, `unsignedinteger` | `isinteger`, `isfloat` | `DType.IsInteger`, `IsFloat`, `IsComplex`, `IsSigned` | partial | abstract scalar hierarchy; the predicates cover it |
| `result_type`, `promote_types` | | `ResultType` | done | checked cell by cell against NumPy 2.5.3 |
| `can_cast`, `min_scalar_type`, `common_type`, `mintypecode`, `typename` | `isa` | | missing | casting rules table (`safe`, `same_kind`) |
| `issubdtype`, `isdtype` | `isa` | | partial | the `DType.Is*` predicates; no kind strings |
| `finfo`, `iinfo` | `realmax`, `realmin`, `eps`, `intmax`, `intmin` | | missing | none (constants from `math`) |
| `ScalarType`, `sctypeDict`, `typecodes` | | | missing | not applicable |

## numpy: printing, errors, I/O, memory, constants, other

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `array_repr` | `disp` | `Repr`, `Display` | done | |
| `array_str` | `disp` | `String` | done | |
| `array2string`, `format_float_positional`, `format_float_scientific`, `base_repr`, `binary_repr`, `get_printoptions`, `set_printoptions`, `printoptions` | `format`, `num2str`, `dec2bin` | | missing | print options struct; float formatting is `strconv` |
| `errstate`, `seterr`, `geterr`, `seterrcall`, `geterrcall`, `getbufsize`, `setbufsize` | | | missing | not applicable (no floating-point error state; operations follow IEEE) |
| `save`, `load` | `save`, `load` | `MarshalBinary`, `UnmarshalBinary` | partial | `.npy` / `.npz` formats (a Go-specific encoding is used); `allow_pickle` is never supported |
| `savez`, `savez_compressed` | `save` | | missing | `.npz` (zip of `.npy`) |
| `savetxt`, `loadtxt`, `genfromtxt` | `writematrix`, `readmatrix`, `dlmwrite` | | missing | none |
| `packbits`, `unpackbits` | | | missing | none |
| `memmap` | `memmapfile` | | missing | `syscall.Mmap` with a build tag |
| `shares_memory`, `may_share_memory`, `isfortran` | | | missing | none (strides analysis) |
| `matrix`, `asmatrix`, `bmat` | | | missing | legacy; not planned |
| `poly`, `poly1d`, `polyadd`, `polyder`, `polydiv`, `polyfit`, `polyint`, `polymul`, `polysub`, `polyval`, `roots` | `poly`, `polyder`, `polyint`, `polyfit`, `polyval`, `roots`, `conv`, `deconv` | | missing | `roots` needs `linalg.Eigvals` (have); `polyfit` needs `Lstsq` (have) |
| `emath` | `sqrt` of negative real, `log` | | missing | complex ufuncs |
| `pi`, `e`, `euler_gamma`, `inf`, `nan` | `pi`, `exp(1)`, `Inf`, `NaN` | `math` constants | done | |
| `newaxis`, `True_`, `False_`, `little_endian` | | `ExpandDims` for `newaxis` | missing | not applicable as values |
| `get_include`, `info`, `show_runtime`, `show_config` | `version`, `ver` | | missing | not applicable |

## numpy.linalg

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `solve` | `\`, `linsolve` | `linalg.Solve` | done | stacked inputs supported |
| `inv` | `inv` | `linalg.Inv` | done | |
| `det` | `det` | `linalg.Det` | done | |
| `slogdet` | | `linalg.Slogdet` | done | |
| `lstsq` | `\`, `lsqminnorm` | `linalg.Lstsq`, `LstsqRcond` | done | |
| `pinv` | `pinv` | `linalg.Pinv`, `PinvRcond` | partial | `hermitian=True` |
| `cholesky` | `chol` | `linalg.Cholesky` | done | `upper` via `UPLO` |
| `qr` | `qr` | `linalg.QR` | done | modes reduced, complete, r, raw per `QRMode` |
| `svd` | `svd` | `linalg.SVD` | partial | `compute_uv=False` (use `SVDVals`), `hermitian=True` |
| `svdvals` | `svd` (values only) | `linalg.SVDVals` | done | |
| `eig` | `eig` | `linalg.Eig` | done | |
| `eigvals` | `eig` (values) | `linalg.Eigvals` | done | |
| `eigh` | `eig` (symmetric) | `linalg.Eigh` | done | |
| `eigvalsh` | | `linalg.Eigvalsh` | done | |
| `norm` | `norm` | `linalg.Norm` | done | |
| `matrix_norm`, `vector_norm` | `norm`, `vecnorm` | `linalg.Norm` with `axes` | partial | named functions and `keepdims` defaults |
| `cond` | `cond` | `linalg.Cond` | done | |
| `matrix_rank` | `rank` | `linalg.MatrixRank`, `MatrixRankTol` | done | |
| `matrix_power` | `^` (matrix), `mpower` | `linalg.MatrixPower` | done | |
| `multi_dot` | | `linalg.MultiDot` | done | |
| `cross` | `cross` | | missing | none |
| `tensordot`, `matmul`, `outer` | `pagemtimes` | `Array.MatMul`, `Outer` | partial | `tensordot` missing; `matmul` 2-D only |
| `tensorsolve`, `tensorinv` | | | missing | `tensordot` / reshape |
| `trace`, `diagonal`, `matrix_transpose` | `trace`, `diag` | | missing | none |
| `LinAlgError` | | `linalg.ErrLinAlg`, `ErrSingular` | done | |

## numpy.random

### Generator (the 1.17+ API)

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `integers` | `randi` | `Generator.Integers`, `IntegersOf`, `IntegersUint64` | done | |
| `random` | `rand` | `Generator.Random`, `RandomOf` | done | |
| `choice` | `randsample` | `Generator.Choice`, `ChoiceN` | done | |
| `bytes` | | `Generator.Bytes` | done | |
| `shuffle`, `permutation`, `permuted` | `randperm` | `Generator.Shuffle`, `Permutation`, `PermutationOf`, `Permuted`, `PermutedAxis` | done | |
| `beta` | `betarnd` | `Generator.Beta` | done | |
| `binomial` | `binornd` | `Generator.Binomial` | done | |
| `chisquare` | `chi2rnd` | `Generator.ChiSquare` | done | |
| `dirichlet` | | `Generator.Dirichlet` | done | |
| `exponential` | `exprnd` | `Generator.Exponential` | done | |
| `f` | `frnd` | `Generator.F` | done | |
| `gamma` | `gamrnd` | `Generator.Gamma` | done | |
| `geometric` | `geornd` | `Generator.Geometric` | done | |
| `gumbel` | | `Generator.Gumbel` | done | |
| `laplace` | | `Generator.Laplace` | done | |
| `logistic` | | `Generator.Logistic` | done | |
| `lognormal` | `lognrnd` | `Generator.Lognormal` | done | |
| `logseries` | | `Generator.Logseries` | done | |
| `multinomial` | `mnrnd` | `Generator.Multinomial` | done | |
| `negative_binomial` | `nbinrnd` | `Generator.NegativeBinomial` | done | |
| `noncentral_chisquare` | `ncx2rnd` | `Generator.NoncentralChiSquare` | done | |
| `noncentral_f` | `ncfrnd` | `Generator.NoncentralF` | done | |
| `normal` | `randn`, `normrnd` | `Generator.Normal` | done | |
| `pareto` | | `Generator.Pareto` | done | |
| `poisson` | `poissrnd` | `Generator.Poisson` | done | |
| `power` | | `Generator.Power` | done | |
| `rayleigh` | `raylrnd` | `Generator.Rayleigh` | done | |
| `standard_cauchy` | | `Generator.StandardCauchy` | done | |
| `standard_exponential` | | `Generator.StandardExponential`, `StandardExponentialOf` | done | |
| `standard_gamma` | | `Generator.StandardGamma`, `StandardGammaOf` | done | |
| `standard_normal` | `randn` | `Generator.StandardNormal`, `StandardNormalOf` | done | |
| `standard_t` | `trnd` | `Generator.StandardT` | done | |
| `triangular` | | `Generator.Triangular` | done | |
| `uniform` | `rand`, `unifrnd` | `Generator.Uniform` | done | |
| `vonmises` | | `Generator.VonMises` | done | |
| `wald` | | `Generator.Wald` | done | |
| `weibull` | `wblrnd` | `Generator.Weibull` | done | |
| `zipf` | | `Generator.Zipf` | done | |
| `spawn`, `bit_generator` | `RandStream` | `Generator.Spawn`, `BitGenerator` | done | |
| `hypergeometric` | `hygernd` | | missing | none (HRUA / HYP algorithms of NumPy for bit-exact streams) |
| `multivariate_hypergeometric` | | | missing | `hypergeometric` |
| `multivariate_normal` | `mvnrnd` | | missing | `linalg.SVD` / `Cholesky` / `Eigh` (have) |

### Bit generators and seeding

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `default_rng`, `Generator` | `rng`, `RandStream` | `DefaultRNG`, `NewGenerator`, `Generator` | done | |
| `BitGenerator` | | `BitGenerator` | done | |
| `MT19937` | `'twister'` | `MT19937` | done | |
| `PCG64`, `PCG64DXSM` | | `PCG64`, `PCG64DXSM` | done | |
| `Philox` | `'philox4x32_10'` | `Philox` | done | |
| `SFC64` | | `SFC64` | done | |
| `SeedSequence` | | `SeedSequence` | done | |
| `RandomState` | `RandStream` | `RandomState` | done | |

### Legacy module-level functions

Done: a `RandomState` method and a package-level function on the global
state. Partial: the `RandomState` method exists, the package-level function
does not.

| NumPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `binomial`, `choice`, `normal`, `permutation`, `poisson`, `rand`, `randint`, `randn`, `random_sample`, `seed`, `shuffle`, `uniform` | `rand`, `randn`, `randi`, `randperm` | package functions `Binomial`, `Choice`, `Normal`, `Permutation`, `Poisson`, `Rand`, `Randint`, `Randn`, `RandomSample`, `Seed`, `Shuffle`, `Uniform` | done | |
| `bytes`, `chisquare`, `exponential`, `gamma`, `get_state`, `lognormal`, `multinomial`, `random`, `set_state`, `standard_exponential`, `standard_gamma`, `standard_normal` | | `RandomState` methods only | partial | package-level function on the global state |
| `beta`, `dirichlet`, `f`, `geometric`, `gumbel`, `laplace`, `logistic`, `logseries`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_t`, `triangular`, `vonmises`, `wald`, `weibull`, `zipf` | | | missing | `RandomState` methods (the legacy streams differ from `Generator`'s; each is a port of `legacy` C code) |
| `hypergeometric`, `multivariate_normal` | | | missing | as in the Generator table |
| `random_integers`, `ranf`, `sample` | | | missing | aliases of `randint` (closed interval), `random_sample` |

## scipy.linalg

| SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| `inv` | `inv` | `linalg.Inv` | done | |
| `solve` | `\` | `linalg.Solve` | partial | `assume_a` (gen, sym, her, pos), `transposed`, `lower`, `check_finite` |
| `solve_banded`, `solveh_banded` | | | missing | banded LAPACK (`gbsv`, `pbsv`); none |
| `solve_circulant`, `solve_toeplitz`, `matmul_toeplitz` | | | missing | FFT (go-fft `fft` subpackage) for circulant; Levinson for Toeplitz |
| `solve_triangular` | `\` on triangular | | missing | `trtrs`; none (the kernel exists inside `linalg`) |
| `det` | `det` | `linalg.Det` | done | |
| `norm` | `norm` | `linalg.Norm` | done | `check_finite` |
| `lstsq` | `\`, `lsqminnorm` | `linalg.Lstsq` | partial | `lapack_driver` choices, `cond` name, `overwrite` flags |
| `pinv` | `pinv` | `linalg.Pinv` | done | |
| `pinvh` | `pinv` | | missing | `Eigh` (have) |
| `khatri_rao` | | | missing | none |
| `orthogonal_procrustes` | | | missing | `SVD` (have) |
| `matrix_balance` | `balance` | | missing | balancing exists inside `Eig`; export |
| `subspace_angles` | `subspace` | | missing | `QR`, `SVD` (have) |
| `bandwidth`, `issymmetric`, `ishermitian` | `bandwidth`, `issymmetric`, `ishermitian` | | missing | none |
| `LinAlgError` | | `linalg.ErrLinAlg` | done | |
| `LinAlgWarning` | | | missing | not applicable (no warnings) |
| `eig` | `eig` | `linalg.Eig` | partial | generalized problem `b=`, left eigenvectors, `homogeneous_eigvals` |
| `eigvals` | `eig` | `linalg.Eigvals` | partial | generalized problem |
| `eigh` | `eig` | `linalg.Eigh` | partial | generalized problem, `subset_by_index`, `subset_by_value`, `driver` |
| `eigvalsh` | | `linalg.Eigvalsh` | partial | same |
| `eig_banded`, `eigvals_banded`, `eigh_tridiagonal`, `eigvalsh_tridiagonal` | | | missing | tridiagonal QL/QR or MRRR / bisection; none |
| `lu`, `lu_factor`, `lu_solve` | `lu`, `decomposition` | | missing | none (LU is inside `Solve`, `Det`; export with pivots and `permute_l`) |
| `svd` | `svd` | `linalg.SVD` | partial | `lapack_driver` (gesdd / gesvd), `compute_uv` |
| `svdvals` | `svd` | `linalg.SVDVals` | done | |
| `diagsvd`, `orth`, `null_space` | `orth`, `null` | | missing | `SVD` (have) |
| `ldl` | `ldl` | | missing | Bunch-Kaufman |
| `cholesky` | `chol` | `linalg.Cholesky` | done | |
| `cholesky_banded`, `cho_factor`, `cho_solve`, `cho_solve_banded` | | | missing | `Cholesky` (have) |
| `polar` | | | missing | `SVD` (have) |
| `qr` | `qr` | `linalg.QR` | partial | `pivoting=True`, `lwork`, `mode` strings |
| `qr_multiply`, `qr_update`, `qr_delete`, `qr_insert` | `qrupdate`, `qrinsert`, `qrdelete` | | missing | Givens rotations; none |
| `rq` | | | missing | `QR` of the reverse |
| `qz`, `ordqz` | `qz`, `ordqz` | | missing | generalized Schur (Hessenberg-triangular, QZ iteration) |
| `schur` | `schur` | | missing | the Schur form is computed inside `Eig`; export, plus `sort` |
| `rsf2csf`, `cdf2rdf` | `rsf2csf`, `cdf2rsf` | | missing | `schur` |
| `hessenberg` | `hess` | | missing | reduction exists inside `Eig`; export |
| `cossin` | | | missing | CS decomposition (`orcsd`) |
| `expm` | `expm` | | missing | Pade with scaling-squaring (Higham), `Solve` (have) |
| `logm` | `logm` | | missing | inverse scaling and squaring, `schur` |
| `sqrtm` | `sqrtm` | | missing | `schur` |
| `cosm`, `sinm`, `tanm`, `coshm`, `sinhm`, `tanhm` | | | missing | `expm` |
| `signm` | | | missing | `eig` / Newton iteration |
| `funm` | `funm` | | missing | `schur` and Parlett recurrence |
| `expm_frechet`, `expm_cond` | | | missing | `expm` |
| `fractional_matrix_power` | `mpower` | | missing | `schur`, `logm`, `expm` |
| `solve_sylvester` | `sylvester` | | missing | `schur` (Bartels-Stewart) |
| `solve_continuous_are`, `solve_discrete_are` | `icare`, `idare` | | missing | `schur`/`qz`, generalized Schur |
| `solve_continuous_lyapunov`, `solve_discrete_lyapunov` | `lyap`, `dlyap` | | missing | `solve_sylvester` |
| `clarkson_woodruff_transform` | | | missing | sparse sketch; `random` (have) |
| `block_diag`, `circulant`, `companion`, `convolution_matrix`, `fiedler`, `fiedler_companion`, `hadamard`, `hankel`, `helmert`, `hilbert`, `invhilbert`, `leslie`, `pascal`, `invpascal`, `toeplitz` | `blkdiag`, `circshift`, `compan`, `hadamard`, `hankel`, `hilb`, `invhilb`, `pascal`, `toeplitz` | | missing | none |
| `dft` | `dftmtx` | | missing | go-fft (have) or direct |
| `get_blas_funcs`, `get_lapack_funcs`, `find_best_blas_type` | | | missing | not applicable (BLAS is internal: `linalg/blas.go`) |
| `scipy.linalg.interpolative` (module: `interp_decomp`, `estimate_rank`, ...) | | | missing | `QR` with pivoting, `SVD`, `random` |
| `scipy.linalg.blas` (level 1-3 routines: `gemm`, `gemv`, `axpy`, ...) | | `MatMul`, `Dot` on the packed GEMM | partial | no public BLAS-named API |
| `scipy.linalg.lapack` (all LAPACK drivers) | | internal routines in `linalg` | partial | a public LAPACK-named API is not planned |

## MATLAB core matrix functions (not already listed above)

MATLAB names with no row in a table above, or whose NumPy name differs
enough that a reader would look them up here.

| MATLAB | NumPy | Go | Status | Missing / dep |
|---|---|---|---|---|
| `repmat`, `repelem` | `tile`, `repeat` | | missing | see shape section |
| `circshift` | `roll` | | missing | |
| `sortrows` | `lexsort`, `argsort` | | missing | sort |
| `cummax`, `cummin` | `maximum.accumulate` | | missing | ufunc accumulate |
| `mode` | `scipy.stats.mode` | | missing | sort, unique |
| `rms`, `normalize`, `rescale`, `bounds` | | | missing | none |
| `movmean`, `movsum`, `movmedian`, `movmax`, `movmin`, `smoothdata` | `sliding_window_view` plus reductions | | missing | strided sliding views |
| `accumarray` | `bincount`, `add.at` | | missing | |
| `bsxfun` | broadcasting | | done | implicit in binary ops |
| `cumtrapz` | `scipy.integrate.cumulative_trapezoid` | | missing | none |
| `del2` | | | missing | `gradient` |
| `nnz` | `count_nonzero` | | missing | none |
| `isempty` | `size == 0` | `Size` | done | |
| `numel`, `length` | `size` | `Size` | done | |
| `colon`, `end` | slicing | `Slice`, `R`, `Rng`, `From`, `To`, `Step` | done | |
| `cat` | `concatenate` | `Concatenate` | done | |
| `isrow`, `iscolumn`, `isvector`, `ismatrix`, `isscalar`, `issquare` | | `Shape`, `Ndim` | done | predicates are one-liners over `Shape` |
| `triplequad`, `integral`, `ode45`, `fzero`, `fminsearch` | `scipy.integrate`, `scipy.optimize` | | missing | go-numerics (`integrate`, `optimize`); not this package |
| `gpuArray`, `tall`, `sparse` | `cupy`, `dask`, `scipy.sparse` | | missing | sparse: a go-sparse package; GPU: not planned |
| `fft`, `ifft`, `fft2` | `numpy.fft` | `fft.FFT`, `fft.IFFT`, `fft.FFT2` | done | bridge to go-fft |
| `conv2`, `filter2`, `imfilter` | `scipy.signal.convolve2d` | | missing | `convolve`; go-fft for large kernels |

## Bridges (this repository's subpackages)

| Name | Go | Status |
|---|---|---|
| `numpy.fft` on arrays: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`, `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftn`, `ifftn`, `rfftn`, `irfftn`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` | package `fft` (over go-fft) with the same names and `Options` | done (18 of 18); see go-fft `docs/coverage.md` for what the engine lacks |
| `numpy.linalg` | package `linalg` | see the numpy.linalg table |
| `numpy.random` | package `random` | see the numpy.random tables |

## Proposed slices

1. **Elementwise math completion** (no dependency): the 40 missing ufuncs
   (inverse trigonometric and hyperbolic, `expm1`, `log1p`, `hypot`,
   `sign`, `floor_divide`, `mod`, `isnan`..., `bitwise_*`, `gcd`, `lcm`),
   one kernel per function with the existing SIMD scaffolding.
2. **Views and shape**: `permute_dims`, `swapaxes`, `moveaxis`, `broadcast_to`,
   `diagonal`, `split` family, `flip`, `roll`, `tile`, `repeat`, `pad`,
   `atleast_nd`, `meshgrid`, `*_like`. No dependency; zero-copy where NumPy is.
3. **Reductions**: multi-axis tuples, `initial`, `where`, `std`, `var`,
   `ptp`, `count_nonzero`, `nan*`, `all`/`any` with axis, `diff`, `gradient`,
   `trapezoid`, `cov`, `corrcoef`. No dependency.
4. **Sort and sets**: `sort`, `argsort`, `partition`, `searchsorted`,
   `unique*`, set operations, `median`, `percentile`, `quantile`,
   `histogram*`, `bincount`, `digitize`. Dependency: a sort kernel.
5. **Batched matmul and contraction**: stacked `matmul`, N-D `dot`, `tensordot`,
   `einsum` (with a path planner), `kron`, `cross`, `trace`, `vdot`,
   `vecdot`. Dependency: the GEMM (have).
6. **scipy.linalg exports of existing internals**: `lu`, `lu_factor`,
   `lu_solve`, `solve_triangular`, `cho_*`, `schur`, `hessenberg`,
   `matrix_balance`, `pinvh`, `orth`, `null_space`, `polar`, `diagsvd`,
   `bandwidth`, `issymmetric`, special matrices. No dependency.
7. **Matrix functions and equations**: `expm`, `logm`, `sqrtm`, `funm`,
   `fractional_matrix_power`, the trigonometric family, `solve_sylvester`,
   Lyapunov, Riccati. Dependency: slice 6 (`schur`).
8. **Banded, tridiagonal, QZ, updates**: `solve_banded`, `eig_banded`,
   `eigh_tridiagonal`, `qz`, `ordqz`, `qr_update` family, `ldl`, `cossin`.
9. **random**: `hypergeometric`, `multivariate_hypergeometric`,
   `multivariate_normal`; legacy `RandomState` distributions and the
   module-level functions. Bit-exact ports of NumPy's C code.
10. **dtypes**: platform alias names, `float16`, then extended precision and
    datetime. float16 needs conversion kernels; datetime and string dtypes are
    a storage-model change and are the decision to take first.
11. **I/O and utilities**: `.npy` / `.npz`, `loadtxt`, `savetxt`, `memmap`,
    print options, `finfo`/`iinfo`, `isclose`, `allclose`, `array_equal`,
    `nan_to_num`, polynomial functions (`polyfit`, `roots`, ...).
12. **User-defined ufuncs and iterators**: `frompyfunc`, `vectorize`,
    `ufunc.reduce/accumulate/outer/at`, `nditer`, `ndindex`.

## Counting

The totals were produced by a small Go program that reads this file and
counts, for each `##` section, the backticked names in the first column by
the Status column. It is not part of the repository (this change is docs
only); recount after editing a row. A second check compares the names of the
NumPy import blocks of `numpy/__init__.py` v2.5.3 with the first columns of
the "numpy:" tables: every name must appear exactly once.
