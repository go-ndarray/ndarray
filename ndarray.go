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
//
// Three layered fast paths avoid the per-element multi-dimensional odometer:
//
//   - Fully contiguous, offset 0: a single bulk copy() of the backing slice.
//   - Unit-stride innermost axis (the common case for row-major slices and any
//     view whose last dimension is untouched): walk only the outer axes with the
//     odometer and copy() each contiguous inner run in one shot. This turns an
//     O(n) per-element gather into O(outer) memmoves of length inner.
//   - Otherwise (e.g. a transposed view, where the last axis is strided): the
//     general per-element gather.
func (a *Array) materialize() []float64 {
	n := a.Size()
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	if a.isContiguous() {
		copy(out, a.data[a.offset:a.offset+n])
		return out
	}
	nd := len(a.shape)
	// Unit-stride innermost axis: bulk-copy contiguous inner runs.
	if nd > 0 && (a.strides[nd-1] == 1 || a.shape[nd-1] == 1) {
		inner := a.shape[nd-1]
		idx := make([]int, nd-1)
		dst := 0
		for dst < n {
			pos := a.offset
			for axis, i := range idx {
				pos += i * a.strides[axis]
			}
			copy(out[dst:dst+inner], a.data[pos:pos+inner])
			dst += inner
			// increment the outer odometer (all axes but the innermost)
			for axis := nd - 2; axis >= 0; axis-- {
				idx[axis]++
				if idx[axis] < a.shape[axis] {
					break
				}
				idx[axis] = 0
			}
		}
		return out
	}
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

// inferReshape resolves a reshape target that may contain a single -1, which is
// inferred so the total size is preserved (NumPy semantics). Other dimensions
// must be non-negative; the inferred dimension must divide the remaining size.
func inferReshape(shape []int, size int) ([]int, error) {
	negAt := -1
	known := 1
	for i, d := range shape {
		switch {
		case d == -1:
			if negAt >= 0 {
				return nil, fmt.Errorf("%w: can only specify one unknown (-1) dimension in %v",
					ErrShapeMismatch, shape)
			}
			negAt = i
		case d < 0:
			return nil, fmt.Errorf("%w: negative dimension %d", ErrShapeMismatch, d)
		default:
			known *= d
		}
	}
	out := append([]int(nil), shape...)
	if negAt < 0 {
		return out, nil
	}
	if known == 0 || size%known != 0 {
		return nil, fmt.Errorf("%w: cannot infer -1 reshaping size %d into %v",
			ErrShapeMismatch, size, shape)
	}
	out[negAt] = size / known
	return out, nil
}

// Reshape returns a view (or copy) of the array with a new shape of the same
// total size. The data is preserved in row-major order. At most one dimension
// may be -1, which is inferred from the total size (NumPy semantics).
func (a *Array) Reshape(shape ...int) (*Array, error) {
	shape, err := inferReshape(shape, a.Size())
	if err != nil {
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
	// Fast path: the innermost axis is materialised with unit stride (it is
	// either a real, non-broadcast last axis or a broadcast length-1 one). Then
	// each contiguous inner run is a single copy() — for a broadcast last axis the
	// source run is one element splatted; for a real last axis it is a memmove of
	// the row. This turns the per-element gather into O(outer) bulk operations,
	// the dominant cost for row/column broadcasts against a large matrix.
	inner := shape[n-1]
	if estrides[n-1] == 1 {
		idx := make([]int, n-1)
		dst := 0
		for dst < len(out) {
			pos := a.offset
			for axis, i := range idx {
				pos += i * estrides[axis]
			}
			copy(out[dst:dst+inner], a.data[pos:pos+inner])
			dst += inner
			for axis := n - 2; axis >= 0; axis-- {
				idx[axis]++
				if idx[axis] < shape[axis] {
					break
				}
				idx[axis] = 0
			}
		}
		return out
	}
	if estrides[n-1] == 0 {
		// Broadcast (length-1) innermost axis: each inner run is a single source
		// element repeated `inner` times — splat it in one tight loop per run.
		idx := make([]int, n-1)
		dst := 0
		for dst < len(out) {
			pos := a.offset
			for axis, i := range idx {
				pos += i * estrides[axis]
			}
			v := a.data[pos]
			run := out[dst : dst+inner]
			for j := range run {
				run[j] = v
			}
			dst += inner
			for axis := n - 2; axis >= 0; axis-- {
				idx[axis]++
				if idx[axis] < shape[axis] {
					break
				}
				idx[axis] = 0
			}
		}
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

// operandFor returns a contiguous []float64 of length prod(shape) holding the
// receiver's elements in the broadcast shape. When the receiver already has
// exactly that shape and is contiguous (the overwhelmingly common same-shape
// case), its backing data is returned directly with no copy — the kernel only
// reads it, so sharing is safe and the per-op allocation drops from three full
// buffers to one (just dst). Otherwise it materialises the broadcast view.
func (a *Array) operandFor(shape []int) []float64 {
	if a.isContiguous() && sameShape(a.shape, shape) {
		return a.data
	}
	return a.broadcastTo(shape)
}

// binOp applies a contiguous-slice kernel to two arrays with NumPy
// broadcasting, returning a new contiguous array of the broadcast shape. The
// kernel must only read its x/y inputs (it may not alias them into dst), since
// operandFor can hand it the operands' live backing slices uncopied.
func (a *Array) binOp(b *Array, kernel func(dst, x, y []float64)) (*Array, error) {
	// Fast path: both operands already contiguous and the same shape (the
	// overwhelmingly common case — `x OP y` on matching arrays). It skips
	// broadcastShape and operandFor entirely (no shape int-slice allocation, no
	// broadcast scan), feeds the live backing slices straight to the kernel, and
	// shares a's shape/strides for the result (read-only; Array is treated as
	// immutable by every op here). Only the result data slice is allocated, so
	// this is the minimal-overhead single-core path that matches NumPy's C loop
	// at small n. Strided/broadcasting operands fall through to the general path.
	if a.isContiguous() && b.isContiguous() && sameShape(a.shape, b.shape) {
		dst := make([]float64, len(a.data))
		kernel(dst, a.data, b.data)
		return &Array{data: dst, shape: a.shape, strides: a.strides}, nil
	}
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	x := a.operandFor(shape)
	y := b.operandFor(shape)
	dst := make([]float64, prod(shape))
	kernel(dst, x, y)
	cp := append([]int(nil), shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}, nil
}

// Add returns the elementwise sum a+b with broadcasting.
func (a *Array) Add(b *Array) (*Array, error) { return a.binOp(b, kernels.AddP) }

// Sub returns the elementwise difference a-b with broadcasting.
func (a *Array) Sub(b *Array) (*Array, error) { return a.binOp(b, kernels.SubP) }

// Mul returns the elementwise product a*b with broadcasting.
func (a *Array) Mul(b *Array) (*Array, error) { return a.binOp(b, kernels.MulP) }

// Div returns the elementwise quotient a/b with broadcasting.
func (a *Array) Div(b *Array) (*Array, error) { return a.binOp(b, kernels.DivP) }

// binOpInto writes a OP b into the caller-provided contiguous out array, the
// no-allocation analogue of binOp and of NumPy's `np.add(a, b, out=z)`. It is
// the parity path for small contiguous ops: NumPy beats Go's allocating Add at
// small n purely because its temp-array free-list makes the result buffer nearly
// free (~60 ns) whereas Go's GC-managed make is ~340 ns; reusing out removes that
// gap entirely (and Go's SIMD kernel then wins outright). out must be contiguous
// and have exactly the broadcast result shape; the operands may be strided or
// broadcasting (they are materialised as needed, but the common contiguous
// same-shape case feeds the live backing slices straight to the kernel). out may
// alias a or b (the kernels read each index before writing it).
func (a *Array) binOpInto(out, b *Array, kernel func(dst, x, y []float64)) error {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return err
	}
	if !out.isContiguous() {
		return fmt.Errorf("%w: out must be contiguous", ErrBroadcast)
	}
	if !sameShape(out.shape, shape) {
		return fmt.Errorf("%w: out shape %v != result shape %v",
			ErrBroadcast, out.shape, shape)
	}
	x := a.operandFor(shape)
	y := b.operandFor(shape)
	kernel(out.data, x, y)
	return nil
}

// AddInto writes a+b into out (no allocation), the parity-path analogue of
// np.add(a, b, out=out). out must be contiguous and the broadcast result shape;
// it may alias a or b. See binOpInto.
func (a *Array) AddInto(out, b *Array) error { return a.binOpInto(out, b, kernels.AddP) }

// SubInto writes a-b into out (no allocation). See AddInto.
func (a *Array) SubInto(out, b *Array) error { return a.binOpInto(out, b, kernels.SubP) }

// MulInto writes a*b into out (no allocation). See AddInto.
func (a *Array) MulInto(out, b *Array) error { return a.binOpInto(out, b, kernels.MulP) }

// DivInto writes a/b into out (no allocation). See AddInto.
func (a *Array) DivInto(out, b *Array) error { return a.binOpInto(out, b, kernels.DivP) }

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

// Map returns a new contiguous array with f applied to every element. For a
// contiguous receiver the elements are read in place (no materialise copy); the
// elementwise pass is parallelised across cores above the kernel threshold. f
// must be safe to call concurrently (the package's math ufuncs are).
func (a *Array) Map(f func(float64) float64) *Array {
	src := a.contiguousData()
	dst := make([]float64, len(src))
	kernels.MapP(dst, src, f)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}
}

// contiguousData returns the receiver's elements as a contiguous row-major
// slice for read-only kernel input, sharing the backing data when the receiver
// is already contiguous and copying (materialising) only for strided views.
func (a *Array) contiguousData() []float64 {
	if a.isContiguous() {
		return a.data
	}
	return a.materialize()
}

// Neg returns the elementwise negation.
func (a *Array) Neg() *Array { return a.Map(func(x float64) float64 { return -x }) }

// Abs returns the elementwise absolute value.
func (a *Array) Abs() *Array { return a.Map(kernels.Abs) }

// --- reductions ------------------------------------------------------------

// Sum returns the sum of all elements (0 for an empty array). Large sums are
// computed as a tree of per-core partials, so the result can differ from a
// strictly left-to-right sum by a few ULP (floating-point addition is not
// associative) — the same trade-off NumPy's pairwise summation makes.
func (a *Array) Sum() float64 { return kernels.SumP(a.contiguousData()) }

// Prod returns the product of all elements (1 for an empty array).
func (a *Array) Prod() float64 { return kernels.Prod(a.contiguousData()) }

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
	return kernels.MaxP(a.contiguousData()), nil
}

// Min returns the minimum element. It returns an error for an empty array.
func (a *Array) Min() (float64, error) {
	if a.Size() == 0 {
		return 0, fmt.Errorf("%w: min of empty array", ErrShapeMismatch)
	}
	return kernels.MinP(a.contiguousData()), nil
}

// --- axis reductions -------------------------------------------------------

// ErrAxis is returned when an axis argument is out of range for the array.
var ErrAxis = errors.New("ndarray: axis out of range")

// normalizeAxis maps a possibly-negative axis to its [0, Ndim) form, matching
// NumPy semantics where axis -1 is the last dimension.
func (a *Array) normalizeAxis(axis int) (int, error) {
	n := len(a.shape)
	if axis < 0 {
		axis += n
	}
	if axis < 0 || axis >= n {
		return 0, fmt.Errorf("%w: %d for %d-dimensional array", ErrAxis, axis, n)
	}
	return axis, nil
}

// reduceShape returns the output shape after reducing the given (normalized)
// axis. With keepdims the reduced axis is retained with length 1; otherwise it
// is removed.
func (a *Array) reduceShape(axis int, keepdims bool) []int {
	if keepdims {
		out := append([]int(nil), a.shape...)
		out[axis] = 1
		return out
	}
	out := make([]int, 0, len(a.shape)-1)
	out = append(out, a.shape[:axis]...)
	out = append(out, a.shape[axis+1:]...)
	return out
}

// reduceLayout splits the shape around a normalized axis into the
// (outer, axisLen, inner) triple used by the axis-reduction kernels, where the
// materialised row-major data is viewed as [outer][axisLen][inner].
func (a *Array) reduceLayout(axis int) (outer, axisLen, inner int) {
	outer = prod(a.shape[:axis])
	axisLen = a.shape[axis]
	inner = prod(a.shape[axis+1:])
	return
}

// reduceAxis is the shared driver for the axis reductions. It validates the
// axis, materialises the data, runs the supplied kernel, and wraps the result
// in an array of the reduced shape. It rejects reduction along a zero-length
// axis (the empty-reduction case), matching this package's whole-array
// reductions which error on empty input.
func (a *Array) reduceAxis(
	axis int, keepdims bool,
	kernel func(dst, src []float64, outer, axisLen, inner int),
) (*Array, error) {
	axis, err := a.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	outer, axisLen, inner := a.reduceLayout(axis)
	if axisLen == 0 {
		return nil, fmt.Errorf("%w: reduction along zero-length axis %d",
			ErrShapeMismatch, axis)
	}
	// contiguousData shares the backing slice when the receiver is already
	// row-major (the common case), so a contiguous array reaches the kernel with
	// zero copy; only strided views pay a materialise. The kernel only reads src.
	src := a.contiguousData()
	dst := make([]float64, outer*inner)
	kernels.RunAxisP(kernel, dst, src, outer, axisLen, inner)
	shape := a.reduceShape(axis, keepdims)
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape)}, nil
}

// SumAxis returns the sum along the given axis. With keepdims the reduced axis
// is kept with length 1 (e.g. (2,3) summed over axis 0 -> (1,3)); otherwise it
// is removed (-> (3,)). A negative axis counts from the end.
func (a *Array) SumAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.SumAxis)
}

// ProdAxis returns the product along the given axis. See SumAxis for the
// axis/keepdims semantics.
func (a *Array) ProdAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.ProdAxis)
}

// MaxAxis returns the maximum along the given axis. See SumAxis for the
// axis/keepdims semantics.
func (a *Array) MaxAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.MaxAxis)
}

// MinAxis returns the minimum along the given axis. See SumAxis for the
// axis/keepdims semantics.
func (a *Array) MinAxis(axis int, keepdims bool) (*Array, error) {
	return a.reduceAxis(axis, keepdims, kernels.MinAxis)
}

// MeanAxis returns the arithmetic mean along the given axis. See SumAxis for the
// axis/keepdims semantics.
func (a *Array) MeanAxis(axis int, keepdims bool) (*Array, error) {
	r, err := a.SumAxis(axis, keepdims)
	if err != nil {
		return nil, err
	}
	// The reduced axis length is positive (SumAxis rejected zero-length axes),
	// so this scaling is well-defined.
	ax, _ := a.normalizeAxis(axis)
	n := float64(a.shape[ax])
	for i := range r.data {
		r.data[i] /= n
	}
	return r, nil
}
