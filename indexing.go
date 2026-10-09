package ndarray

import "fmt"

// Boolean / fancy indexing.
//
// A mask is normally a Bool array (what the comparison ufuncs Greater/Equal/…
// return), but as in NumPy any dtype is accepted and "truthy" means non-zero,
// so the 0/1 float masks of earlier versions keep working. MaskSelect is
// a[mask], Nonzero is np.flatnonzero(mask), and Take is integer fancy indexing
// into the flat array.

// MaskSelect returns a 1-D array of the elements of a where mask is truthy
// (non-zero), in row-major order: NumPy's a[mask] for a mask of a's shape, the
// usual case (a mask from a.Greater(...)). mask may also broadcast to a's shape,
// which NumPy rejects (IndexError): a (3,) mask selects the same columns of
// every row of a (2, 3). Two NumPy forms differ: a mask covering only the
// leading axes, a[rowmask], is rejected here unless it broadcasts; and a 0-d
// mask selects all or none of a, flattened, where NumPy returns a with a new
// leading axis of length 1 or 0.
func (a *Array) MaskSelect(mask *Array) (*Array, error) {
	shape, err := broadcastShape(a.shape, mask.shape)
	if err != nil {
		return nil, err
	}
	if !sameShape(shape, a.shape) {
		// NumPy requires the boolean index to match the indexed array's shape
		// (after broadcasting it must not enlarge a); reject a mask that would.
		return nil, fmt.Errorf("%w: mask shape %v does not match array shape %v",
			ErrShapeMismatch, mask.shape, a.shape)
	}
	if a.dtype != Float64 || mask.dtype != Float64 {
		mb := mask.operandStore(Bool, shape).([]bool)
		var idx []int
		for i, m := range mb {
			if m {
				idx = append(idx, i)
			}
		}
		return a.result(takeStore(a.operandStore(a.dtype, shape), idx), []int{len(idx)}), nil
	}
	av := a.operandFor(shape)
	mv := mask.operandFor(shape)
	out := make([]float64, 0)
	for i, m := range mv {
		if m != 0 {
			out = append(out, av[i])
		}
	}
	return &Array{data: out, shape: []int{len(out)}, strides: []int{1}, ws: a.ws}, nil
}

// Nonzero returns a 1-D Int64 array of the flat (row-major) indices at which
// a is non-zero — NumPy's np.flatnonzero(a). For a mask this is the positions
// of the true elements.
func (a *Array) Nonzero() *Array {
	b := a.operandStore(Bool, a.shape).([]bool)
	out := make([]int64, 0)
	for i, v := range b {
		if v {
			out = append(out, int64(i))
		}
	}
	return a.result(out, []int{len(out)})
}

// Take returns a 1-D array gathering a's flattened (row-major) elements at the
// given integer indices — NumPy's a.take(indices) / fancy indexing a[idx].
// Negative indices count from the end; an out-of-range index errors.
func (a *Array) Take(indices ...int) (*Array, error) {
	if a.dtype != Float64 {
		n := a.Size()
		idx := make([]int, len(indices))
		for k, i := range indices {
			j := i
			if j < 0 {
				j += n
			}
			if j < 0 || j >= n {
				return nil, fmt.Errorf("%w: take index %d out of range for size %d",
					ErrShapeMismatch, i, n)
			}
			idx[k] = j
		}
		return a.result(takeStore(a.contiguousStore(), idx), []int{len(idx)}), nil
	}
	flat := a.contiguousData()
	n := len(flat)
	out := a.alloc(len(indices), false)
	for k, idx := range indices {
		j := idx
		if j < 0 {
			j += n
		}
		if j < 0 || j >= n {
			return nil, fmt.Errorf("%w: take index %d out of range for size %d",
				ErrShapeMismatch, idx, n)
		}
		out[k] = flat[j]
	}
	return &Array{data: out, shape: []int{len(out)}, strides: []int{1}, ws: a.ws}, nil
}
