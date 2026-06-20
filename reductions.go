package ndarray

import (
	"fmt"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// ArgMax returns the flat index of the first maximum element over the array in
// row-major order, matching numpy.argmax with no axis. It errors on an empty
// array.
func (a *Array) ArgMax() (int, error) {
	if a.Size() == 0 {
		return 0, fmt.Errorf("%w: argmax of empty array", ErrShapeMismatch)
	}
	return kernels.ArgMax(a.contiguousData()), nil
}

// ArgMin returns the flat index of the first minimum element over the array in
// row-major order, matching numpy.argmin with no axis. It errors on an empty
// array.
func (a *Array) ArgMin() (int, error) {
	if a.Size() == 0 {
		return 0, fmt.Errorf("%w: argmin of empty array", ErrShapeMismatch)
	}
	return kernels.ArgMin(a.contiguousData()), nil
}

// ArgMaxAxis returns the indices of the first maxima along the given axis,
// matching numpy.argmax(axis=...). The result has the reduced shape (the axis
// removed, or kept as length 1 with keepdims) and integer index values stored
// as float64. A negative axis counts from the end.
func (a *Array) ArgMaxAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.ArgMaxAxis)
}

// ArgMinAxis returns the indices of the first minima along the given axis,
// matching numpy.argmin(axis=...). See ArgMaxAxis for the shape semantics.
func (a *Array) ArgMinAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.ArgMinAxis)
}

// scanAxis is the shared driver for the cumulative scans (CumSum/CumProd along
// an axis): it validates the axis, materialises the data, runs the supplied
// scan kernel over the [outer][axisLen][inner] layout, and returns a new array
// of the SAME shape as the input (scans do not reduce). A zero-length axis is
// allowed and yields an equal-shaped empty result.
func (a *Array) scanAxis(
	axis int,
	kernel func(dst, src []float64, outer, axisLen, inner int),
) (*Array, error) {
	axis, err := a.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	outer, axisLen, inner := a.reduceLayout(axis)
	src := a.materialize()
	dst := make([]float64, len(src))
	kernel(dst, src, outer, axisLen, inner)
	shape := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape)}, nil
}

// CumSum returns the cumulative sum along the given axis, matching
// numpy.cumsum(axis=...). The result has the same shape as the input. A
// negative axis counts from the end.
func (a *Array) CumSum(axis int) (*Array, error) {
	return a.scanAxis(axis, kernels.CumSumAxis)
}

// CumProd returns the cumulative product along the given axis, matching
// numpy.cumprod(axis=...). The result has the same shape as the input.
func (a *Array) CumProd(axis int) (*Array, error) {
	return a.scanAxis(axis, kernels.CumProdAxis)
}

// CumSumFlat returns the cumulative sum over the flattened array (1-D, row-major
// order), matching numpy.cumsum with no axis.
func (a *Array) CumSumFlat() *Array {
	src := a.materialize()
	dst := make([]float64, len(src))
	kernels.CumSumAxis(dst, src, 1, len(src), 1)
	return &Array{data: dst, shape: []int{len(dst)}, strides: []int{1}}
}

// CumProdFlat returns the cumulative product over the flattened array (1-D,
// row-major order), matching numpy.cumprod with no axis.
func (a *Array) CumProdFlat() *Array {
	src := a.materialize()
	dst := make([]float64, len(src))
	kernels.CumProdAxis(dst, src, 1, len(src), 1)
	return &Array{data: dst, shape: []int{len(dst)}, strides: []int{1}}
}

// Clip returns a new array with every element limited to the range [lo, hi],
// matching numpy.clip. It errors if lo > hi.
func (a *Array) Clip(lo, hi float64) (*Array, error) {
	if lo > hi {
		return nil, fmt.Errorf("%w: clip bounds lo=%g > hi=%g", ErrShapeMismatch, lo, hi)
	}
	src := a.contiguousData()
	dst := make([]float64, len(src))
	kernels.Clip(dst, src, lo, hi)
	shape := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape)}, nil
}

// Where returns an array selecting from t where cond is truthy (non-zero) and
// from f otherwise, elementwise with full NumPy broadcasting across all three
// operands — matching numpy.where(cond, t, f). cond is typically a 0/1 mask
// from the comparison ufuncs.
func Where(cond, t, f *Array) (*Array, error) {
	shape, err := broadcastShape(cond.shape, t.shape)
	if err != nil {
		return nil, err
	}
	shape, err = broadcastShape(shape, f.shape)
	if err != nil {
		return nil, err
	}
	c := cond.operandFor(shape)
	tv := t.operandFor(shape)
	fv := f.operandFor(shape)
	dst := make([]float64, prod(shape))
	kernels.Where(dst, c, tv, fv)
	cp := append([]int(nil), shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}, nil
}
