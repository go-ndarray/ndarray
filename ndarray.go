// Package ndarray is a pure-Go (CGO=0) NumPy-style N-dimensional array library.
//
// The element type for this phase is float64. The numeric kernels live in
// internal/kernels behind a contiguous-slice API so that go-asmgen SIMD kernels
// (amd64, arm64, riscv64, loong64, ppc64le, s390x) can replace them in a later
// phase without touching the public API. See docs/plan-ndarray.md for the
// roadmap.
//
// Arrays are stored row-major (C-order). An Array is a view over a flat data
// slice described by a shape, per-axis strides (in elements) and a base offset,
// so reshaping and transposing can be zero-copy where possible.
package ndarray

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// Array is an N-dimensional, row-major array of float64.
type Array struct {
	data    []float64
	shape   []int
	strides []int
	offset  int
}

// ErrShapeMismatch is returned when a requested shape is invalid or
// incompatible with the data it must describe.
var ErrShapeMismatch = errors.New("ndarray: shape mismatch")

// ErrBroadcast is returned when two shapes cannot be broadcast together.
var ErrBroadcast = errors.New("ndarray: shapes are not broadcastable")

// prod returns the product of a list of dimensions; the product of an empty
// list (a scalar / 0-d array) is 1.
func prod(dims []int) int {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return n
}

// validateShape reports whether every dimension is non-negative.
func validateShape(shape []int) error {
	for _, d := range shape {
		if d < 0 {
			return fmt.Errorf("%w: negative dimension %d", ErrShapeMismatch, d)
		}
	}
	return nil
}

// rowMajorStrides returns the C-order strides for shape.
func rowMajorStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

// New allocates a zero-filled array with the given shape.
func New(shape ...int) (*Array, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	cp := append([]int(nil), shape...)
	return &Array{
		data:    make([]float64, prod(cp)),
		shape:   cp,
		strides: rowMajorStrides(cp),
	}, nil
}

// Zeros is an alias for New: a zero-filled array of the given shape.
func Zeros(shape ...int) (*Array, error) {
	return New(shape...)
}

// Ones returns an array of the given shape filled with 1.
func Ones(shape ...int) (*Array, error) {
	return Full(1, shape...)
}

// Full returns an array of the given shape filled with v.
func Full(v float64, shape ...int) (*Array, error) {
	a, err := New(shape...)
	if err != nil {
		return nil, err
	}
	for i := range a.data {
		a.data[i] = v
	}
	return a, nil
}

// FromData wraps data in an array of the given shape. The data slice is copied,
// so later mutations of the caller's slice do not affect the array. The number
// of elements implied by shape must equal len(data).
func FromData(data []float64, shape ...int) (*Array, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	if prod(shape) != len(data) {
		return nil, fmt.Errorf("%w: shape %v implies %d elements, got %d",
			ErrShapeMismatch, shape, prod(shape), len(data))
	}
	cp := append([]int(nil), shape...)
	return &Array{
		data:    append([]float64(nil), data...),
		shape:   cp,
		strides: rowMajorStrides(cp),
	}, nil
}

// Arange returns a 1-D array of evenly spaced values over [start, stop) with the
// given step, matching numpy.arange. step must be non-zero.
func Arange(start, stop, step float64) (*Array, error) {
	if step == 0 {
		return nil, fmt.Errorf("%w: step must be non-zero", ErrShapeMismatch)
	}
	var data []float64
	if step > 0 {
		for v := start; v < stop; v += step {
			data = append(data, v)
		}
	} else {
		for v := start; v > stop; v += step {
			data = append(data, v)
		}
	}
	return FromData(data, len(data))
}

// Shape returns a copy of the array's shape.
func (a *Array) Shape() []int {
	return append([]int(nil), a.shape...)
}

// Ndim returns the number of dimensions.
func (a *Array) Ndim() int {
	return len(a.shape)
}

// Size returns the total number of elements.
func (a *Array) Size() int {
	return prod(a.shape)
}

// flatIndex maps a multi-dimensional index to a position in the data slice,
// honouring offset and strides. It panics on rank or bounds errors, matching
// Go slice-indexing conventions.
func (a *Array) flatIndex(idx []int) int {
	if len(idx) != len(a.shape) {
		panic(fmt.Sprintf("ndarray: got %d indices for %d-dimensional array",
			len(idx), len(a.shape)))
	}
	pos := a.offset
	for axis, i := range idx {
		if i < 0 || i >= a.shape[axis] {
			panic(fmt.Sprintf("ndarray: index %d out of range for axis %d with size %d",
				i, axis, a.shape[axis]))
		}
		pos += i * a.strides[axis]
	}
	return pos
}

// At returns the element at the given multi-dimensional index.
func (a *Array) At(idx ...int) float64 {
	return a.data[a.flatIndex(idx)]
}

// Set stores v at the given multi-dimensional index.
func (a *Array) Set(v float64, idx ...int) {
	a.data[a.flatIndex(idx)] = v
}

// iterIndices visits every multi-dimensional index of shape in row-major order
// and calls f with the flat data position (relative to offset/strides) and the
// linear count. It is the workhorse for materialising views.
func (a *Array) forEach(f func(linear, pos int)) {
	n := a.Size()
	if n == 0 {
		return
	}
	idx := make([]int, len(a.shape))
	for linear := 0; linear < n; linear++ {
		pos := a.offset
		for axis, i := range idx {
			pos += i * a.strides[axis]
		}
		f(linear, pos)
		// increment row-major odometer
		for axis := len(idx) - 1; axis >= 0; axis-- {
			idx[axis]++
			if idx[axis] < a.shape[axis] {
				break
			}
			idx[axis] = 0
		}
	}
}

// materialize returns a fresh, contiguous, offset-0 copy of the array's data in
// row-major order. The result is independent of the receiver's storage.
func (a *Array) materialize() []float64 {
	out := make([]float64, a.Size())
	a.forEach(func(linear, pos int) {
		out[linear] = a.data[pos]
	})
	return out
}

// isContiguous reports whether the array is a row-major contiguous view with no
// offset, so its data slice can be used directly.
func (a *Array) isContiguous() bool {
	if a.offset != 0 {
		return false
	}
	expect := rowMajorStrides(a.shape)
	for i := range expect {
		// A dimension of length 1 has an arbitrary stride; ignore it.
		if a.shape[i] != 1 && a.strides[i] != expect[i] {
			return false
		}
	}
	return a.Size() == len(a.data)
}

// Copy returns a deep, contiguous copy of the array.
func (a *Array) Copy() *Array {
	cp := append([]int(nil), a.shape...)
	return &Array{
		data:    a.materialize(),
		shape:   cp,
		strides: rowMajorStrides(cp),
	}
}

// Reshape returns a view (or copy) of the array with a new shape of the same
// total size. The data is preserved in row-major order.
func (a *Array) Reshape(shape ...int) (*Array, error) {
	if err := validateShape(shape); err != nil {
		return nil, err
	}
	if prod(shape) != a.Size() {
		return nil, fmt.Errorf("%w: cannot reshape size %d into %v",
			ErrShapeMismatch, a.Size(), shape)
	}
	cp := append([]int(nil), shape...)
	if a.isContiguous() {
		return &Array{
			data:    a.data,
			shape:   cp,
			strides: rowMajorStrides(cp),
		}, nil
	}
	return &Array{
		data:    a.materialize(),
		shape:   cp,
		strides: rowMajorStrides(cp),
	}, nil
}

// Ravel returns a contiguous 1-D array containing the elements in row-major
// order.
func (a *Array) Ravel() *Array {
	data := a.materialize()
	return &Array{
		data:    data,
		shape:   []int{len(data)},
		strides: []int{1},
	}
}

// Transpose returns a view with the axes reversed. No data is copied; only the
// shape and strides are permuted.
func (a *Array) Transpose() *Array {
	n := len(a.shape)
	shape := make([]int, n)
	strides := make([]int, n)
	for i := 0; i < n; i++ {
		shape[i] = a.shape[n-1-i]
		strides[i] = a.strides[n-1-i]
	}
	return &Array{
		data:    a.data,
		shape:   shape,
		strides: strides,
		offset:  a.offset,
	}
}

// String renders the array's shape and its elements in row-major order.
func (a *Array) String() string {
	var b strings.Builder
	b.WriteString("Array(shape=[")
	for i, d := range a.shape {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(strconv.Itoa(d))
	}
	b.WriteString("], data=[")
	flat := a.materialize()
	for i, v := range flat {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(strconv.FormatFloat(v, 'g', -1, 64))
	}
	b.WriteString("])")
	return b.String()
}

// --- broadcasting ----------------------------------------------------------

// broadcastShape computes the broadcast result shape of two shapes per the
// NumPy rules: align trailing dimensions; each pair must be equal or one of
// them must be 1.
func broadcastShape(s1, s2 []int) ([]int, error) {
	n := len(s1)
	if len(s2) > n {
		n = len(s2)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		d1, d2 := 1, 1
		if j := len(s1) - n + i; j >= 0 {
			d1 = s1[j]
		}
		if j := len(s2) - n + i; j >= 0 {
			d2 = s2[j]
		}
		switch {
		case d1 == d2:
			out[i] = d1
		case d1 == 1:
			out[i] = d2
		case d2 == 1:
			out[i] = d1
		default:
			return nil, fmt.Errorf("%w: %v vs %v", ErrBroadcast, s1, s2)
		}
	}
	return out, nil
}

// broadcastTo returns a contiguous []float64 of length prod(shape) holding the
// receiver's elements broadcast to shape. shape must be broadcast-compatible
// with the receiver (it is, by construction, the broadcast result shape).
func (a *Array) broadcastTo(shape []int) []float64 {
	n := len(shape)
	// Build effective shape/strides aligned to n trailing dims; a missing or
	// length-1 dimension gets stride 0 so it repeats.
	estrides := make([]int, n)
	for i := 0; i < n; i++ {
		j := len(a.shape) - n + i
		if j < 0 || a.shape[j] == 1 {
			estrides[i] = 0
		} else {
			estrides[i] = a.strides[j]
		}
	}
	out := make([]float64, prod(shape))
	if len(out) == 0 {
		return out
	}
	idx := make([]int, n)
	for linear := range out {
		pos := a.offset
		for axis, i := range idx {
			pos += i * estrides[axis]
		}
		out[linear] = a.data[pos]
		for axis := n - 1; axis >= 0; axis-- {
			idx[axis]++
			if idx[axis] < shape[axis] {
				break
			}
			idx[axis] = 0
		}
	}
	return out
}

// binOp applies a contiguous-slice kernel to two arrays with NumPy
// broadcasting, returning a new contiguous array of the broadcast shape.
func (a *Array) binOp(b *Array, kernel func(dst, x, y []float64)) (*Array, error) {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	x := a.broadcastTo(shape)
	y := b.broadcastTo(shape)
	dst := make([]float64, len(x))
	kernel(dst, x, y)
	cp := append([]int(nil), shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}, nil
}

// Add returns the elementwise sum a+b with broadcasting.
func (a *Array) Add(b *Array) (*Array, error) { return a.binOp(b, kernels.Add) }

// Sub returns the elementwise difference a-b with broadcasting.
func (a *Array) Sub(b *Array) (*Array, error) { return a.binOp(b, kernels.Sub) }

// Mul returns the elementwise product a*b with broadcasting.
func (a *Array) Mul(b *Array) (*Array, error) { return a.binOp(b, kernels.Mul) }

// Div returns the elementwise quotient a/b with broadcasting.
func (a *Array) Div(b *Array) (*Array, error) { return a.binOp(b, kernels.Div) }

// scalarArray wraps a scalar as a 0-d array so it broadcasts against anything.
func scalarArray(v float64) *Array {
	return &Array{data: []float64{v}, shape: []int{}, strides: []int{}}
}

// AddScalar returns the array with v added to every element.
func (a *Array) AddScalar(v float64) *Array { r, _ := a.Add(scalarArray(v)); return r }

// SubScalar returns the array with v subtracted from every element.
func (a *Array) SubScalar(v float64) *Array { r, _ := a.Sub(scalarArray(v)); return r }

// MulScalar returns the array with every element multiplied by v.
func (a *Array) MulScalar(v float64) *Array { r, _ := a.Mul(scalarArray(v)); return r }

// DivScalar returns the array with every element divided by v.
func (a *Array) DivScalar(v float64) *Array { r, _ := a.Div(scalarArray(v)); return r }

// --- unary / map -----------------------------------------------------------

// Map returns a new contiguous array with f applied to every element.
func (a *Array) Map(f func(float64) float64) *Array {
	src := a.materialize()
	dst := make([]float64, len(src))
	kernels.Map(dst, src, f)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}
}

// Neg returns the elementwise negation.
func (a *Array) Neg() *Array { return a.Map(func(x float64) float64 { return -x }) }

// Abs returns the elementwise absolute value.
func (a *Array) Abs() *Array { return a.Map(kernels.Abs) }

// --- reductions ------------------------------------------------------------

// Sum returns the sum of all elements (0 for an empty array).
func (a *Array) Sum() float64 { return kernels.Sum(a.materialize()) }

// Prod returns the product of all elements (1 for an empty array).
func (a *Array) Prod() float64 { return kernels.Prod(a.materialize()) }

// Mean returns the arithmetic mean of all elements. It returns an error for an
// empty array.
func (a *Array) Mean() (float64, error) {
	n := a.Size()
	if n == 0 {
		return 0, fmt.Errorf("%w: mean of empty array", ErrShapeMismatch)
	}
	return a.Sum() / float64(n), nil
}

// Max returns the maximum element. It returns an error for an empty array.
func (a *Array) Max() (float64, error) {
	if a.Size() == 0 {
		return 0, fmt.Errorf("%w: max of empty array", ErrShapeMismatch)
	}
	return kernels.Max(a.materialize()), nil
}

// Min returns the minimum element. It returns an error for an empty array.
func (a *Array) Min() (float64, error) {
	if a.Size() == 0 {
		return 0, fmt.Errorf("%w: min of empty array", ErrShapeMismatch)
	}
	return kernels.Min(a.materialize()), nil
}
