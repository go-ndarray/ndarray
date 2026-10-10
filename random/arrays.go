package random

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/go-ndarray/ndarray"
)

// ErrValue is wrapped by every error that numpy raises as ValueError (or
// TypeError) for a bad argument; the message after it is numpy's.
var ErrValue = errors.New("random")

// ErrNotContiguous is returned by the in-place Shuffle for an array that is
// a strided view: it has no storage of its own to permute.
var ErrNotContiguous = errors.New("random: shuffle needs an array that owns contiguous storage (call Copy first)")

func valueErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrValue}, args...)...)
}

// elem is the element types of ndarray's dtypes.
type elem interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 |
		float32 | float64 | complex64 | complex128
}

// newArray returns a fresh zero array of dtype dt (whose element type is T)
// and its storage.
func newArray[T elem](dt ndarray.DType, shape []int) (*ndarray.Array, []T, error) {
	a, err := ndarray.ZerosOf(dt, shape...)
	if err != nil {
		return nil, nil, err
	}
	d, _ := ndarray.Data[T](a)
	return a, d, nil
}

// storage returns a's elements as a contiguous []T in an any; the array's
// own storage when it is contiguous, else a copy.
func storage(a *ndarray.Array) any {
	switch a.DType() {
	case ndarray.Bool:
		d, _ := ndarray.Data[bool](a)
		return d
	case ndarray.Int8:
		d, _ := ndarray.Data[int8](a)
		return d
	case ndarray.Int16:
		d, _ := ndarray.Data[int16](a)
		return d
	case ndarray.Int32:
		d, _ := ndarray.Data[int32](a)
		return d
	case ndarray.Int64:
		d, _ := ndarray.Data[int64](a)
		return d
	case ndarray.Uint8:
		d, _ := ndarray.Data[uint8](a)
		return d
	case ndarray.Uint16:
		d, _ := ndarray.Data[uint16](a)
		return d
	case ndarray.Uint32:
		d, _ := ndarray.Data[uint32](a)
		return d
	case ndarray.Uint64:
		d, _ := ndarray.Data[uint64](a)
		return d
	case ndarray.Float32:
		d, _ := ndarray.Data[float32](a)
		return d
	case ndarray.Complex64:
		d, _ := ndarray.Data[complex64](a)
		return d
	case ndarray.Complex128:
		d, _ := ndarray.Data[complex128](a)
		return d
	}
	d, _ := ndarray.Data[float64](a)
	return d
}

// ownStorage returns a's own contiguous storage, or ErrNotContiguous when a
// is a view (Data then hands out a fresh copy on every call).
func ownStorage(a *ndarray.Array) (reflect.Value, error) {
	s := reflect.ValueOf(storage(a))
	if s.Len() > 0 && s.Pointer() != reflect.ValueOf(storage(a)).Pointer() {
		return s, ErrNotContiguous
	}
	return s, nil
}

// arrayOf returns a fresh array of a's dtype and the given shape, and its
// storage.
func arrayOf(dt ndarray.DType, shape []int) (*ndarray.Array, reflect.Value, error) {
	a, err := ndarray.ZerosOf(dt, shape...)
	if err != nil {
		return nil, reflect.Value{}, err
	}
	return a, reflect.ValueOf(storage(a)), nil
}

// normAxis is numpy's normalize_axis_index.
func normAxis(axis, ndim int) (int, error) {
	if axis < -ndim || axis >= ndim {
		return 0, valueErr("axis %d is out of bounds for array of dimension %d", axis, ndim)
	}
	if axis < 0 {
		axis += ndim
	}
	return axis, nil
}

// layout splits shape around axis into (outer, n, inner) element counts.
func layout(shape []int, axis int) (outer, n, inner int) {
	outer, inner = 1, 1
	for _, d := range shape[:axis] {
		outer *= d
	}
	for _, d := range shape[axis+1:] {
		inner *= d
	}
	return outer, shape[axis], inner
}

// takeAxis is numpy's a.take(idx, axis): the result has a's shape with the
// axis replaced by idxShape. Indices are already known to be in range.
func takeAxis(a *ndarray.Array, idx []int64, idxShape []int, axis int) (*ndarray.Array, error) {
	shape := a.Shape()
	outer, n, inner := layout(shape, axis)
	rs := append(append(append([]int(nil), shape[:axis]...), idxShape...), shape[axis+1:]...)
	src := reflect.ValueOf(storage(a))
	r, dst, err := arrayOf(a.DType(), rs)
	if err != nil {
		return nil, err
	}
	pos := 0
	for o := range outer {
		for _, k := range idx {
			from := (o*n + int(k)) * inner
			reflect.Copy(dst.Slice(pos, pos+inner), src.Slice(from, from+inner))
			pos += inner
		}
	}
	return r, nil
}

// swapSlabs swaps positions i and j along the middle axis of the (outer, n,
// inner) layout of s.
func swapSlabs(swap func(i, j int), outer, n, inner, i, j int) {
	for o := range outer {
		bi := (o*n + i) * inner
		bj := (o*n + j) * inner
		for k := range inner {
			swap(bi+k, bj+k)
		}
	}
}
