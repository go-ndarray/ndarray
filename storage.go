package ndarray

import (
	"fmt"
	"math"
	"strconv"
)

//go:generate go run gen_dtypes.go

// Element type sets. They list exact types, not ~T: storage is always one of
// these slices, and the type switches that dispatch on it name them exactly.
type (
	intT interface {
		int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
	}
	floatT   interface{ float32 | float64 }
	complexT interface{ complex64 | complex128 }
	realT    interface{ intT | floatT }
	numT     interface{ realT | complexT }
	elemT    interface{ numT | bool }
)

// dtypeOf returns the DType whose storage is []T.
func dtypeOf[T elemT]() DType {
	var z T
	switch any(z).(type) {
	case bool:
		return Bool
	case int8:
		return Int8
	case int16:
		return Int16
	case int32:
		return Int32
	case int64:
		return Int64
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	case float32:
		return Float32
	case complex64:
		return Complex64
	case complex128:
		return Complex128
	}
	return Float64
}

// store returns a's backing storage as an untyped slice: a.data for Float64,
// a.ext for every other dtype.
func (a *Array) store() any {
	if a.dtype == Float64 {
		return a.data
	}
	return a.ext
}

// typed returns a's backing storage as a []T. T must be a's element type.
func typed[T elemT](a *Array) []T { return a.store().([]T) }

// fromStore wraps a contiguous storage slice (one of the element slices) in a
// row-major array of the given shape, which it takes ownership of.
func fromStore(s any, shape []int) *Array {
	a := &Array{shape: shape, strides: rowMajorStrides(shape)}
	if d, ok := s.([]float64); ok {
		a.data = d
		return a
	}
	a.ext = s
	a.dtype = dtypeOfStore(s)
	return a
}

// view returns an array sharing a's storage, dtype and workspace with the
// given layout.
func (a *Array) view(shape, strides []int, offset int) *Array {
	return &Array{data: a.data, ext: a.ext, dtype: a.dtype,
		shape: shape, strides: strides, offset: offset, ws: a.ws}
}

// result wraps a fresh contiguous storage slice computed from a: it is
// shaped like shape and bound to a's workspace.
func (a *Array) result(s any, shape []int) *Array {
	r := fromStore(s, shape)
	r.ws = a.ws
	return r
}

// DType returns the array's element type.
func (a *Array) DType() DType { return a.dtype }

// ZerosOf returns a zero-filled array of the given dtype and shape.
func ZerosOf(dt DType, shape ...int) (*Array, error) {
	if dt == Float64 {
		return New(shape...)
	}
	if int(dt) >= numDTypes {
		return nil, fmt.Errorf("%w: %v", ErrDType, dt)
	}
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	cp := append([]int(nil), shape...)
	return fromStore(makeStore(dt, prod(cp)), cp), nil
}

// OnesOf returns an array of the given dtype and shape filled with one.
func OnesOf(dt DType, shape ...int) (*Array, error) { return FullOf(dt, 1, shape...) }

// FullOf returns an array of the given dtype and shape filled with v,
// converted to the dtype as AsType converts.
func FullOf(dt DType, v complex128, shape ...int) (*Array, error) {
	a, err := ZerosOf(dt, shape...)
	if err != nil {
		return nil, err
	}
	src := []complex128{v}
	one := makeStore(dt, 1)
	convertStore(one, src)
	fillStore(a.store(), one)
	return a, nil
}

// FromSlice wraps a copy of data in an array of the given shape. The dtype is
// the one whose element type is T. It is the typed counterpart of FromData.
func FromSlice[T elemT](data []T, shape ...int) (*Array, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	if prod(shape) != len(data) {
		return nil, fmt.Errorf("%w: shape %v implies %d elements, got %d",
			ErrShapeMismatch, shape, prod(shape), len(data))
	}
	cp := append([]int(nil), shape...)
	return fromStore(append([]T(nil), data...), cp), nil
}

// Data returns the array's elements as a contiguous row-major []T. T must be
// the element type of the array's dtype (int8 for Int8, complex128 for
// Complex128 ...); otherwise ok is false. The slice is the array's own storage
// when the array is contiguous and a copy otherwise, so write through it only
// to an array you made.
func Data[T elemT](a *Array) (data []T, ok bool) {
	if a.dtype != dtypeOf[T]() {
		return nil, false
	}
	return a.contiguousStore().([]T), true
}

// Item returns the element at the given index as a T, which must be the
// element type of the array's dtype. It panics like At on a bad index, and on
// a T of another dtype.
func Item[T elemT](a *Array, idx ...int) T {
	if a.dtype != dtypeOf[T]() {
		panic(fmt.Sprintf("ndarray: Item[%v] of a %v array", dtypeOf[T](), a.dtype))
	}
	return typed[T](a)[a.flatIndex(idx)]
}

// AtComplex returns the element at the given index as a complex128, for an
// array of any dtype.
func (a *Array) AtComplex(idx ...int) complex128 {
	return elemComplex(a.store(), a.flatIndex(idx))
}

// SetComplex stores v at the given index, converted to the array's dtype as
// AsType converts (a real dtype keeps the real part).
func (a *Array) SetComplex(v complex128, idx ...int) {
	setComplex(a.store(), a.flatIndex(idx), v)
}

// AsType returns a contiguous copy of the array converted to dtype dt, like
// NumPy's astype. Conversions follow Go's, with these definitions where Go
// leaves the result to the platform or NumPy to the hardware:
//
//   - float to integer truncates toward zero and saturates: values beyond the
//     type's range, and infinities, give its minimum or maximum, and NaN
//     gives 0. NumPy leaves these casts undefined and they differ by
//     platform (on arm64 it saturates to 64 bits and then wraps, so +Inf to
//     int8 is -1; on x86 NaN gives the minimum); here they are the same
//     everywhere;
//   - complex to real keeps the real part (NumPy warns, Go cannot);
//   - anything to Bool is "not zero", so NaN is true.
func (a *Array) AsType(dt DType) *Array {
	src := a.contiguousStore()
	if dt == a.dtype {
		return a.result(cloneStore(src), append([]int(nil), a.shape...))
	}
	dst := makeStore(dt, a.Size())
	convertStore(dst, src)
	return a.result(dst, append([]int(nil), a.shape...))
}

// contiguousStore returns a's elements as a contiguous row-major storage
// slice, sharing the backing storage when a is contiguous.
func (a *Array) contiguousStore() any {
	if a.dtype == Float64 {
		return a.contiguousData()
	}
	if a.isContiguous() {
		return a.ext
	}
	return gatherStore(a)
}

// gather returns a fresh contiguous row-major copy of the elements of the
// strided view (data, shape, strides, offset).
func gather[T any](data []T, shape, strides []int, offset int) []T {
	n := prod(shape)
	out := make([]T, n)
	if n == 0 {
		return out
	}
	idx := make([]int, len(shape))
	for linear := range out {
		pos := offset
		for axis, i := range idx {
			pos += i * strides[axis]
		}
		out[linear] = data[pos]
		for axis := len(idx) - 1; axis >= 0; axis-- {
			idx[axis]++
			if idx[axis] < shape[axis] {
				break
			}
			idx[axis] = 0
		}
	}
	return out
}

// broadcastStrides returns a's strides aligned to the trailing dimensions of
// shape, with 0 for every axis a lacks or has length 1 on, so that indexing
// with them repeats a across shape.
func (a *Array) broadcastStrides(shape []int) []int {
	n := len(shape)
	es := make([]int, n)
	for i := range es {
		if j := len(a.shape) - n + i; j >= 0 && a.shape[j] != 1 {
			es[i] = a.strides[j]
		}
	}
	return es
}

// operandStore returns a's elements, converted to dtype dt, as a contiguous
// storage slice of the broadcast shape. It shares a's storage when nothing
// needs to change.
func (a *Array) operandStore(dt DType, shape []int) any {
	var s any
	if a.isContiguous() && sameShape(a.shape, shape) {
		s = a.store()
	} else {
		s = broadcastStore(a, shape)
	}
	if a.dtype == dt {
		return s
	}
	out := makeStore(dt, prod(shape))
	convertStore(out, s)
	return out
}

// Conversions. The generated convertStore calls these for the cases Go does
// not define or NumPy defines otherwise.

// satInt converts f to the integer type I, truncating toward zero and
// saturating at I's range; NaN gives 0.
func satInt[I intT](f float64) I {
	if f != f {
		return 0
	}
	var lo, hi I
	hi = ^hi
	if hi < 0 { // signed: ^0 is -1
		bits := uint(8 * sizeOf[I]())
		hi = I(uint64(1)<<(bits-1) - 1)
		lo = -hi - 1
	}
	if f <= float64(lo) {
		return lo
	}
	// float64(hi) can round up to 2^bits, one past hi: compare with >=.
	if f >= float64(hi) {
		return hi
	}
	return I(f)
}

// sizeOf returns the size of I in bytes.
func sizeOf[I intT]() int {
	var z I
	switch any(z).(type) {
	case int8, uint8:
		return 1
	case int16, uint16:
		return 2
	case int32, uint32:
		return 4
	}
	return 8
}

// boolTo converts a bool to 1 or 0 of a numeric type.
func boolTo[T numT](b bool) T {
	if b {
		return 1
	}
	return 0
}

// complexString formats a complex number as NumPy does: (1+2j).
func complexString(c complex128, bits int) string {
	re, im := real(c), imag(c)
	sign := "+"
	if im < 0 || (im == 0 && math.Signbit(im)) {
		sign, im = "-", -im
	}
	return "(" + floatString(re, bits) + sign + floatString(im, bits) + "j)"
}

// floatString formats a float as the package always has: the shortest
// representation that reads back to the same value at the given precision.
func floatString(f float64, bits int) string { return strconv.FormatFloat(f, 'g', -1, bits) }
