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
	if a.dtype != Float64 {
		r, _ := a.Ravel().argTyped(0, false, true) // non-empty 1-D: cannot fail
		return int(r.ext.([]int64)[0]), nil
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
	if a.dtype != Float64 {
		r, _ := a.Ravel().argTyped(0, false, false) // non-empty 1-D: cannot fail
		return int(r.ext.([]int64)[0]), nil
	}
	return kernels.ArgMin(a.contiguousData()), nil
}

// ArgMaxAxis returns the indices of the first maxima along the given axis,
// matching numpy.argmax(axis=...). The result is an Int64 array of the
// reduced shape (the axis removed, or kept as length 1 with keepdims). A
// negative axis counts from the end.
func (a *Array) ArgMaxAxis(axis int, keepdims bool) (*Array, error) {
	return a.argTyped(axis, keepdims, true)
}

// ArgMinAxis returns the indices of the first minima along the given axis,
// matching numpy.argmin(axis=...). See ArgMaxAxis for the shape semantics.
func (a *Array) ArgMinAxis(axis int, keepdims bool) (*Array, error) {
	return a.argTyped(axis, keepdims, false)
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
	dst := a.alloc(len(src), false)
	// An empty array can still have a huge axis, (0, 2^58), and the kernel
	// walks outer*axisLen*inner: skip it rather than loop over nothing.
	if len(dst) > 0 {
		kernel(dst, src, outer, axisLen, inner)
	}
	shape := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape), ws: a.ws}, nil
}

// CumSum returns the cumulative sum along the given axis, matching
// numpy.cumsum(axis=...). The result has the same shape as the input. A
// negative axis counts from the end.
func (a *Array) CumSum(axis int) (*Array, error) {
	if a.dtype != Float64 {
		return a.scanTyped(axis, rSum)
	}
	return a.scanAxis(axis, kernels.CumSumAxis)
}

// CumProd returns the cumulative product along the given axis, matching
// numpy.cumprod(axis=...). The result has the same shape as the input.
func (a *Array) CumProd(axis int) (*Array, error) {
	if a.dtype != Float64 {
		return a.scanTyped(axis, rProd)
	}
	return a.scanAxis(axis, kernels.CumProdAxis)
}

// CumSumFlat returns the cumulative sum over the flattened array (1-D, row-major
// order), matching numpy.cumsum with no axis.
func (a *Array) CumSumFlat() *Array {
	if a.dtype != Float64 {
		r, _ := a.Ravel().scanTyped(0, rSum) // 1-D over axis 0: cannot fail
		return r
	}
	src := a.materialize()
	dst := a.alloc(len(src), false)
	kernels.CumSumAxis(dst, src, 1, len(src), 1)
	return &Array{data: dst, shape: []int{len(dst)}, strides: []int{1}, ws: a.ws}
}

// CumProdFlat returns the cumulative product over the flattened array (1-D,
// row-major order), matching numpy.cumprod with no axis.
func (a *Array) CumProdFlat() *Array {
	if a.dtype != Float64 {
		r, _ := a.Ravel().scanTyped(0, rProd) // 1-D over axis 0: cannot fail
		return r
	}
	src := a.materialize()
	dst := a.alloc(len(src), false)
	kernels.CumProdAxis(dst, src, 1, len(src), 1)
	return &Array{data: dst, shape: []int{len(dst)}, strides: []int{1}, ws: a.ws}
}

// Clip returns a new array with every element limited to the range [lo, hi],
// matching numpy.clip. It errors if lo > hi.
func (a *Array) Clip(lo, hi float64) (*Array, error) {
	if lo > hi {
		return nil, fmt.Errorf("%w: clip bounds lo=%g > hi=%g", ErrShapeMismatch, lo, hi)
	}
	if a.dtype != Float64 {
		return a.clipTyped(lo, hi)
	}
	src := a.contiguousData()
	dst := a.alloc(len(src), false)
	kernels.Clip(dst, src, lo, hi)
	shape := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape), ws: a.ws}, nil
}

// Where returns an array selecting from t where cond is truthy (non-zero) and
// from f otherwise, elementwise with full NumPy broadcasting across all three
// operands — matching numpy.where(cond, t, f). cond is typically a Bool mask
// from the comparison ufuncs; any dtype is read by truthiness. The result has
// the common dtype of t and f.
func Where(cond, t, f *Array) (*Array, error) {
	shape, err := broadcastShape(cond.shape, t.shape)
	if err != nil {
		return nil, err
	}
	shape, err = broadcastShape(shape, f.shape)
	if err != nil {
		return nil, err
	}
	if cond.dtype != Float64 || t.dtype != Float64 || f.dtype != Float64 {
		return whereTyped(cond, t, f, shape), nil
	}
	c := cond.operandFor(shape)
	tv := t.operandFor(shape)
	fv := f.operandFor(shape)
	dst := wsOf(cond, t, f).alloc(prod(shape), false)
	kernels.Where(dst, c, tv, fv)
	cp := append([]int(nil), shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp), ws: wsOf(cond, t, f)}, nil
}
