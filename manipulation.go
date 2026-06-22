package ndarray

import "fmt"

// Flatten returns a contiguous 1-D copy of the array's elements in row-major
// order. Unlike Ravel, the result is always an independent copy (NumPy
// flatten semantics; here Ravel is also a copy, but Flatten documents the
// always-copy contract).
func (a *Array) Flatten() *Array {
	data := a.materialize()
	return &Array{data: data, shape: []int{len(data)}, strides: []int{1}}
}

// ExpandDims returns a view with a new length-1 axis inserted at the given
// position, matching numpy.expand_dims. axis may range over [-(Ndim+1), Ndim];
// a negative axis counts from the end of the result.
func (a *Array) ExpandDims(axis int) (*Array, error) {
	n := len(a.shape)
	if axis < 0 {
		axis += n + 1
	}
	if axis < 0 || axis > n {
		return nil, fmt.Errorf("%w: %d for %d-dimensional array", ErrAxis, axis, n)
	}
	shape := make([]int, 0, n+1)
	strides := make([]int, 0, n+1)
	shape = append(shape, a.shape[:axis]...)
	shape = append(shape, 1)
	shape = append(shape, a.shape[axis:]...)
	strides = append(strides, a.strides[:axis]...)
	// The inserted axis has length 1; its stride is irrelevant, use 0.
	strides = append(strides, 0)
	strides = append(strides, a.strides[axis:]...)
	return &Array{data: a.data, shape: shape, strides: strides, offset: a.offset}, nil
}

// Squeeze returns a view with length-1 axes removed. With no axes given, every
// length-1 axis is removed; otherwise only the listed axes are removed and each
// must have length 1 (NumPy semantics). Negative axes count from the end.
func (a *Array) Squeeze(axes ...int) (*Array, error) {
	n := len(a.shape)
	remove := make([]bool, n)
	if len(axes) == 0 {
		for i, d := range a.shape {
			if d == 1 {
				remove[i] = true
			}
		}
	} else {
		for _, ax := range axes {
			norm := ax
			if norm < 0 {
				norm += n
			}
			if norm < 0 || norm >= n {
				return nil, fmt.Errorf("%w: %d for %d-dimensional array", ErrAxis, ax, n)
			}
			if a.shape[norm] != 1 {
				return nil, fmt.Errorf("%w: cannot squeeze axis %d with length %d",
					ErrShapeMismatch, norm, a.shape[norm])
			}
			remove[norm] = true
		}
	}
	shape := make([]int, 0, n)
	strides := make([]int, 0, n)
	for i := 0; i < n; i++ {
		if !remove[i] {
			shape = append(shape, a.shape[i])
			strides = append(strides, a.strides[i])
		}
	}
	return &Array{data: a.data, shape: shape, strides: strides, offset: a.offset}, nil
}

// Concatenate joins the given arrays along an existing axis, matching
// numpy.concatenate. Every array must have the same rank and the same shape on
// every axis except the concatenation axis; a negative axis counts from the
// end. At least one array is required. The result is a fresh contiguous array.
func Concatenate(arrays []*Array, axis int) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("%w: concatenate needs at least one array",
			ErrShapeMismatch)
	}
	first := arrays[0]
	ndim := len(first.shape)
	ax, err := first.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	// The result shape equals the first array's, with the axis lengths summed.
	out := append([]int(nil), first.shape...)
	axisTotal := 0
	for _, arr := range arrays {
		if len(arr.shape) != ndim {
			return nil, fmt.Errorf("%w: concatenate rank %d vs %d",
				ErrShapeMismatch, len(arr.shape), ndim)
		}
		for d := 0; d < ndim; d++ {
			if d != ax && arr.shape[d] != first.shape[d] {
				return nil, fmt.Errorf("%w: concatenate shapes %v vs %v differ on axis %d",
					ErrShapeMismatch, arr.shape, first.shape, d)
			}
		}
		axisTotal += arr.shape[ax]
	}
	out[ax] = axisTotal
	return concatInto(arrays, ax, out), nil
}

// concatInto materialises arrays joined along ax into a contiguous result of
// the given shape. The result is laid out as [outer][axisTotal][inner]; each
// source contributes its axis slab into the matching outer rows.
func concatInto(arrays []*Array, ax int, out []int) *Array {
	inner := prod(out[ax+1:])
	outer := prod(out[:ax])
	axisTotal := out[ax]
	data := make([]float64, outer*axisTotal*inner)
	// Column offset (in the joined axis) where the next array's slab begins.
	colBase := 0
	for _, arr := range arrays {
		// contiguousData shares the backing slice for already-contiguous sources
		// (no copy); the per-outer-row copy() below then moves each axis slab in
		// one bulk memmove rather than element by element.
		src := arr.contiguousData()
		aLen := arr.shape[ax]
		for o := 0; o < outer; o++ {
			dstRow := (o*axisTotal + colBase) * inner
			srcRow := o * aLen * inner
			copy(data[dstRow:dstRow+aLen*inner], src[srcRow:srcRow+aLen*inner])
		}
		colBase += aLen
	}
	return &Array{data: data, shape: out, strides: rowMajorStrides(out)}
}

// Stack joins the given arrays along a new axis, matching numpy.stack. Every
// array must have identical shape; the new axis is inserted at the given
// position, which may range over [-(Ndim+1), Ndim]. The result has rank
// Ndim+1.
func Stack(arrays []*Array, axis int) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("%w: stack needs at least one array", ErrShapeMismatch)
	}
	first := arrays[0]
	for _, arr := range arrays {
		if !sameShape(arr.shape, first.shape) {
			return nil, fmt.Errorf("%w: stack requires identical shapes, %v vs %v",
				ErrShapeMismatch, arr.shape, first.shape)
		}
	}
	// Expand each operand with a length-1 axis at the insertion point, then
	// concatenate along it. ExpandDims validates the axis range.
	expanded := make([]*Array, len(arrays))
	for i, arr := range arrays {
		e, err := arr.ExpandDims(axis)
		if err != nil {
			return nil, err
		}
		expanded[i] = e
	}
	// The concatenation axis is the (normalised) insertion point.
	norm := axis
	if norm < 0 {
		norm += len(first.shape) + 1
	}
	return Concatenate(expanded, norm)
}

// sameShape reports whether two shapes are identical.
func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// atLeast2D promotes a 1-D array of length n to a (1, n) row view for stacking,
// leaving higher-rank arrays unchanged. Used by VStack. A 1-D array is already
// contiguous along its single axis, so the row view is built directly (no
// reshape that could fail).
func atLeast2D(a *Array) *Array {
	if len(a.shape) == 1 {
		return &Array{
			data:    a.data,
			shape:   []int{1, a.shape[0]},
			strides: []int{a.shape[0] * a.strides[0], a.strides[0]},
			offset:  a.offset,
		}
	}
	return a
}

// VStack stacks arrays vertically (row-wise), matching numpy.vstack: 1-D inputs
// of length n are treated as (1, n) rows, then all inputs are concatenated
// along axis 0.
func VStack(arrays []*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("%w: vstack needs at least one array", ErrShapeMismatch)
	}
	promoted := make([]*Array, len(arrays))
	for i, arr := range arrays {
		promoted[i] = atLeast2D(arr)
	}
	return Concatenate(promoted, 0)
}

// HStack stacks arrays horizontally (column-wise), matching numpy.hstack:
// concatenation along axis 1, except that 1-D inputs are joined along axis 0
// (their only axis).
func HStack(arrays []*Array) (*Array, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("%w: hstack needs at least one array", ErrShapeMismatch)
	}
	if len(arrays[0].shape) == 1 {
		return Concatenate(arrays, 0)
	}
	return Concatenate(arrays, 1)
}
