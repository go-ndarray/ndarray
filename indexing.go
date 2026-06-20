package ndarray

import "fmt"

// Boolean / fancy indexing.
//
// A dedicated bool dtype is a later phase; until then a boolean mask is a 0/1
// float array (exactly what the comparison ufuncs Greater/Equal/… return), and
// "truthy" means != 0. These operations bring NumPy boolean indexing to that
// representation: MaskSelect is a[mask], Nonzero is np.flatnonzero(mask), and
// Take is integer fancy indexing into the flat array.

// MaskSelect returns a 1-D array of the elements of a where mask is truthy
// (non-zero), in row-major order — NumPy's a[mask]. mask must broadcast to a's
// shape (typically it has exactly a's shape, e.g. a mask from a.Greater(...)).
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
	av := a.operandFor(shape)
	mv := mask.operandFor(shape)
	out := make([]float64, 0)
	for i, m := range mv {
		if m != 0 {
			out = append(out, av[i])
		}
	}
	return &Array{data: out, shape: []int{len(out)}, strides: []int{1}}, nil
}

// Nonzero returns a 1-D array of the flat (row-major) indices at which a is
// non-zero — NumPy's np.flatnonzero(a). For a 0/1 mask this is the positions of
// the truthy elements.
func (a *Array) Nonzero() *Array {
	av := a.contiguousData()
	out := make([]float64, 0)
	for i, v := range av {
		if v != 0 {
			out = append(out, float64(i))
		}
	}
	return &Array{data: out, shape: []int{len(out)}, strides: []int{1}}
}

// Take returns a 1-D array gathering a's flattened (row-major) elements at the
// given integer indices — NumPy's a.take(indices) / fancy indexing a[idx].
// Negative indices count from the end; an out-of-range index errors.
func (a *Array) Take(indices ...int) (*Array, error) {
	flat := a.contiguousData()
	n := len(flat)
	out := make([]float64, len(indices))
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
	return &Array{data: out, shape: []int{len(out)}, strides: []int{1}}, nil
}
