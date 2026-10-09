package ndarray

import (
	"fmt"
	"math/cmplx"
	"reflect"
)

// The operations on dtypes other than Float64. Each public method keeps its
// Float64 path as it was and comes here when an operand has another dtype;
// when the result is Float64 anyway the operands are converted and the
// Float64 path runs, so its kernels and their speed are kept.

// Scalar returns a 0-d array holding v, to combine with arrays. A Go int,
// float or complex (of any size) is "weak", as a Python scalar is in NumPy 2:
// it takes the other operand's dtype when that can hold it, so
//
//	int8Array.Add(Scalar(1))      // int8
//	float32Array.Mul(Scalar(0.5)) // float32
//	int8Array.Add(Scalar(0.5))    // float64
//
// A bool gives a Bool array. Scalar panics on any other type.
func Scalar(v any) *Array {
	rv := reflect.ValueOf(v)
	var a *Array
	switch rv.Kind() {
	case reflect.Bool:
		a = fromStore([]bool{rv.Bool()}, []int{})
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		a = fromStore([]int64{rv.Int()}, []int{})
		a.weak = weakInt
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		a = fromStore([]uint64{rv.Uint()}, []int{})
		a.weak = weakInt
	case reflect.Float32, reflect.Float64:
		a = scalarArray(rv.Float())
	case reflect.Complex64, reflect.Complex128:
		a = fromStore([]complex128{rv.Complex()}, []int{})
		a.weak = weakComplex
	default:
		panic(fmt.Sprintf("ndarray: Scalar of a %T", v))
	}
	return a
}

// toFloat64 returns a as a Float64 array: a itself, or a converted copy.
func (a *Array) toFloat64() *Array {
	if a.dtype == Float64 {
		return a
	}
	r := a.AsType(Float64)
	r.weak = a.weak
	return r
}

// arith is the binary arithmetic driver for operands of any dtype. k is the
// Float64 kernel, used whenever the result is Float64.
func (a *Array) arith(b *Array, op binCode, k func(dst, x, y []float64)) (*Array, error) {
	if a.dtype == Float64 && b.dtype == Float64 {
		return a.binOp(b, k)
	}
	dt := promote(a, b)
	switch {
	case op == opDiv && (dt == Bool || dt.IsInteger()):
		dt = Float64 // true division
	case op == opSub && dt == Bool:
		return nil, fmt.Errorf("%w: subtracting bool arrays; use LogicalXor", ErrDType)
	}
	if dt == Float64 {
		return a.toFloat64().binOp(b.toFloat64(), k)
	}
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	d := makeStore(dt, prod(shape))
	arithStore(op, d, a.operandStore(dt, shape), b.operandStore(dt, shape))
	return a.result(d, append([]int(nil), shape...)), nil
}

// arithInto is arith writing into out, which must be contiguous and have the
// broadcast shape; the result is converted to out's dtype.
func (a *Array) arithInto(out, b *Array, op binCode, k func(dst, x, y []float64)) error {
	if a.dtype == Float64 && b.dtype == Float64 && out.dtype == Float64 {
		return a.binOpInto(out, b, k)
	}
	r, err := a.arith(b, op, k)
	if err != nil {
		return err
	}
	if !out.isContiguous() {
		return fmt.Errorf("%w: out must be contiguous", ErrBroadcast)
	}
	if !sameShape(out.shape, r.shape) {
		return fmt.Errorf("%w: out shape %v != result shape %v", ErrBroadcast, out.shape, r.shape)
	}
	convertStore(out.store(), r.store())
	return nil
}

// compare is the comparison driver: the operands are promoted to their common
// dtype and compared into a Bool array.
func (a *Array) compare(b *Array, op cmpCode) (*Array, error) {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	dt := promote(a, b)
	d := make([]bool, prod(shape))
	cmpStore(op, d, a.operandStore(dt, shape), b.operandStore(dt, shape))
	return a.result(d, append([]int(nil), shape...)), nil
}

// LogicalNot returns the elementwise "not zero is false" of a, as a Bool array.
func (a *Array) LogicalNot() *Array {
	b := a.operandStore(Bool, a.shape).([]bool)
	d := make([]bool, len(b))
	for i, v := range b {
		d[i] = !v
	}
	return a.result(d, append([]int(nil), a.shape...))
}

// logical is the driver for LogicalAnd/Or/Xor: both operands are read by
// truthiness.
func (a *Array) logical(b *Array, f func(x, y bool) bool) (*Array, error) {
	shape, err := broadcastShape(a.shape, b.shape)
	if err != nil {
		return nil, err
	}
	x := a.operandStore(Bool, shape).([]bool)
	y := b.operandStore(Bool, shape).([]bool)
	d := make([]bool, len(x))
	for i := range d {
		d[i] = f(x[i], y[i])
	}
	return a.result(d, append([]int(nil), shape...)), nil
}

// LogicalAnd returns the elementwise "a and b" of the operands' truthiness,
// with broadcasting, as a Bool array.
func (a *Array) LogicalAnd(b *Array) (*Array, error) {
	return a.logical(b, func(x, y bool) bool { return x && y })
}

// LogicalOr returns the elementwise "a or b" of the operands' truthiness.
func (a *Array) LogicalOr(b *Array) (*Array, error) {
	return a.logical(b, func(x, y bool) bool { return x || y })
}

// LogicalXor returns the elementwise "a xor b" of the operands' truthiness.
func (a *Array) LogicalXor(b *Array) (*Array, error) {
	return a.logical(b, func(x, y bool) bool { return x != y })
}

// Any reports whether any element is non-zero.
func (a *Array) Any() bool {
	for _, v := range a.operandStore(Bool, a.shape).([]bool) {
		if v {
			return true
		}
	}
	return false
}

// All reports whether every element is non-zero (true for an empty array).
func (a *Array) All() bool {
	for _, v := range a.operandStore(Bool, a.shape).([]bool) {
		if !v {
			return false
		}
	}
	return true
}

// floatDType is the dtype NumPy's float ufuncs (sqrt, exp, sin ...) give for an
// input of dtype dt: inexact types are kept; integers become the smallest float
// that holds them exactly. NumPy gives float16 for 8-bit integers and bool;
// there is no float16 here, so they get float32.
func floatDType(dt DType) DType {
	switch {
	case dt.IsFloat() || dt.IsComplex():
		return dt
	case dt.Size() <= 2:
		return Float32
	}
	return Float64
}

// mathUnary applies a float ufunc to a non-Float64 array. f64 is the Float64
// method, used when the result is Float64.
func (a *Array) mathUnary(code unaryCode, f64 func(*Array) *Array) *Array {
	dt := floatDType(a.dtype)
	if dt == Float64 {
		return f64(a.AsType(Float64))
	}
	src := a.operandStore(dt, a.shape)
	d := makeStore(dt, a.Size())
	switch s := src.(type) {
	case []float32:
		unaryFloat32(realFuncs[code], d.([]float32), s)
	case []complex64:
		unaryComplex(a.complexFunc(code), d.([]complex64), s)
	case []complex128:
		unaryComplex(a.complexFunc(code), d.([]complex128), s)
	}
	return a.result(d, append([]int(nil), a.shape...))
}

// complexFunc returns the complex function for a unary code, panicking for an
// operation NumPy does not define on complex numbers.
func (a *Array) complexFunc(code unaryCode) func(complex128) complex128 {
	f := complexFuncs[code]
	if f == nil {
		panic(fmt.Sprintf("ndarray: operation not defined for %v", a.dtype))
	}
	return f
}

// exactUnary is the driver of Floor, Ceil and Round: integers and bools are
// returned unchanged (as a copy), as in NumPy.
func (a *Array) exactUnary(code unaryCode, f64 func(*Array) *Array) *Array {
	if a.dtype == Bool || a.dtype.IsInteger() {
		return a.Copy()
	}
	return a.mathUnary(code, f64)
}

// intUnary applies negation, absolute value or squaring, which keep the dtype
// of an integer array; f64 is the Float64 method.
func (a *Array) intUnary(code unaryCode, f64 func(*Array) *Array) *Array {
	switch {
	case a.dtype == Bool && code == uNeg:
		panic("ndarray: negating a bool array; use LogicalNot")
	case a.dtype == Bool && code == uAbs:
		return a.Copy()
	case a.dtype == Bool: // square of bool is int8, as in NumPy
		return a.AsType(Int8).intUnary(code, f64)
	case a.dtype.IsInteger():
		src := a.contiguousStore()
		d := makeStore(a.dtype, a.Size())
		intUnaryStore(code, d, src)
		return a.result(d, append([]int(nil), a.shape...))
	case a.dtype.IsComplex() && code == uAbs:
		return a.absComplex()
	}
	return a.mathUnary(code, f64)
}

// absComplex is the modulus of a complex array, as a float array of the same
// precision.
func (a *Array) absComplex() *Array {
	n := a.Size()
	var d any
	switch s := a.contiguousStore().(type) {
	case []complex64:
		r := make([]float32, n)
		absComplex(r, s)
		d = r
	case []complex128:
		r := make([]float64, n)
		absComplex(r, s)
		d = r
	}
	return a.result(d, append([]int(nil), a.shape...))
}

// power is Power for a non-Float64 array: the exponent is a weak float.
func (a *Array) power(p float64) *Array {
	src := a.contiguousStore()
	switch s := src.(type) {
	case []float32:
		d := make([]float32, len(s))
		powFloat32(p, d, s)
		return a.result(d, append([]int(nil), a.shape...))
	case []complex64:
		d := make([]complex64, len(s))
		powComplex(complex(p, 0), d, s)
		return a.result(d, append([]int(nil), a.shape...))
	case []complex128:
		d := make([]complex128, len(s))
		powComplex(complex(p, 0), d, s)
		return a.result(d, append([]int(nil), a.shape...))
	}
	return a.AsType(Float64).Power(p)
}

// realOnly panics for a complex array in an operation whose result is a
// float64, naming what to use instead.
func (a *Array) realOnly(op, instead string) {
	if a.dtype.IsComplex() {
		panic(fmt.Sprintf("ndarray: %s of a %v array; use %s", op, a.dtype, instead))
	}
}

// accDType is the dtype NumPy accumulates sums and products of dt in: Int64
// for bools and signed integers, Uint64 for unsigned, dt itself otherwise.
func accDType(dt DType) DType {
	switch {
	case dt == Bool || dt.IsSigned():
		return Int64
	case dt.IsInteger():
		return Uint64
	}
	return dt
}

// reduceTyped is the axis-reduction driver for non-Float64 arrays.
func (a *Array) reduceTyped(axis int, keepdims bool, op reduceCode) (*Array, error) {
	ax, err := a.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	outer, axisLen, inner := a.reduceLayout(ax)
	dt := a.dtype
	if op == rSum || op == rProd {
		dt = accDType(dt)
	}
	shape := a.reduceShape(ax, keepdims)
	dst := makeStore(dt, outer*inner)
	if axisLen == 0 {
		if op == rMax || op == rMin {
			return nil, fmt.Errorf("%w: reduction along zero-length axis %d", ErrShapeMismatch, ax)
		}
		if op == rProd {
			one := makeStore(dt, 1)
			setComplex(one, 0, 1)
			fillStore(dst, one)
		}
		return a.result(dst, shape), nil
	}
	if outer*inner > 0 {
		src := a.operandStore(dt, a.shape)
		if op == rSum || op == rProd {
			accumulateStore(op, dst, src, outer, axisLen, inner)
		} else {
			extremeStore(op, dst, src, outer, axisLen, inner)
		}
	}
	return a.result(dst, shape), nil
}

// reduceAll reduces every element of a non-Float64 array into a 0-d array.
func (a *Array) reduceAll(op reduceCode) (*Array, error) {
	return a.Ravel().reduceTyped(0, false, op)
}

// SumAll returns the sum of all elements as a 0-d array of the dtype NumPy
// gives it: Int64 for bools and signed integers, Uint64 for unsigned, the
// array's own dtype otherwise. Unlike Sum, it holds every dtype exactly,
// including the complex ones and integers beyond 2^53.
func (a *Array) SumAll() *Array {
	if a.dtype == Float64 {
		return scalarOf(a.Sum(), a.ws)
	}
	r, _ := a.reduceAll(rSum) // a 1-D reduction over axis 0 cannot fail
	return r
}

// ProdAll returns the product of all elements as a 0-d array; see SumAll.
func (a *Array) ProdAll() *Array {
	if a.dtype == Float64 {
		return scalarOf(a.Prod(), a.ws)
	}
	r, _ := a.reduceAll(rProd)
	return r
}

// scalarOf is a strong 0-d Float64 array.
func scalarOf(v float64, ws *Workspace) *Array {
	return &Array{data: []float64{v}, shape: []int{}, strides: []int{}, ws: ws}
}

// argTyped is the arg-reduction driver for every dtype: it returns Int64
// indices, as NumPy does.
func (a *Array) argTyped(axis int, keepdims, larger bool) (*Array, error) {
	ax, err := a.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	outer, axisLen, inner := a.reduceLayout(ax)
	if axisLen == 0 {
		return nil, fmt.Errorf("%w: reduction along zero-length axis %d", ErrShapeMismatch, ax)
	}
	dst := make([]int64, outer*inner)
	if len(dst) > 0 {
		argStore(larger, dst, a.contiguousStore(), outer, axisLen, inner)
	}
	return a.result(dst, a.reduceShape(ax, keepdims)), nil
}

// scanTyped is the cumulative-scan driver for non-Float64 arrays.
func (a *Array) scanTyped(axis int, op reduceCode) (*Array, error) {
	ax, err := a.normalizeAxis(axis)
	if err != nil {
		return nil, err
	}
	outer, axisLen, inner := a.reduceLayout(ax)
	dt := accDType(a.dtype)
	dst := makeStore(dt, a.Size())
	if storeLen(dst) > 0 {
		scanStore(op, dst, a.operandStore(dt, a.shape), outer, axisLen, inner)
	}
	return a.result(dst, append([]int(nil), a.shape...)), nil
}

// clipTyped is Clip for a non-Float64 array. The bounds are weak floats: an
// integer array is clipped as float64, a float32 array stays float32.
func (a *Array) clipTyped(lo, hi float64) (*Array, error) {
	switch {
	case a.dtype.IsComplex():
		return nil, fmt.Errorf("%w: clip of a %v array", ErrDType, a.dtype)
	case a.dtype != Float32:
		return a.AsType(Float64).Clip(lo, hi)
	}
	d := make([]float32, a.Size())
	clipOrdered(d, a.contiguousStore().([]float32), float32(lo), float32(hi))
	return a.result(d, append([]int(nil), a.shape...)), nil
}

// whereTyped is Where for operands of any dtype.
func whereTyped(cond, t, f *Array, shape []int) *Array {
	dt := promote(t, f)
	c := cond.operandStore(Bool, shape).([]bool)
	d := makeStore(dt, len(c))
	whereStore(d, c, t.operandStore(dt, shape), f.operandStore(dt, shape))
	r := fromStore(d, append([]int(nil), shape...))
	r.ws = wsOf(cond, t, f)
	return r
}

// Real returns the real part of the elements: a float array of the same
// precision for a complex array, a copy otherwise (as numpy.real).
func (a *Array) Real() *Array {
	if !a.dtype.IsComplex() {
		return a.Copy()
	}
	return a.AsType(complexPart(a.dtype))
}

// Imag returns the imaginary part of the elements: a float array of the same
// precision for a complex array, zeros of a's dtype otherwise (as numpy.imag).
func (a *Array) Imag() *Array {
	if !a.dtype.IsComplex() {
		z, _ := ZerosOf(a.dtype, a.shape...) // a's shape is valid
		z.ws = a.ws
		return z
	}
	r := a.complexMap(func(z complex128) complex128 { return complex(imag(z), 0) })
	return r.AsType(complexPart(a.dtype))
}

// Conj returns the complex conjugate (a copy for a real array).
func (a *Array) Conj() *Array {
	if !a.dtype.IsComplex() {
		return a.Copy()
	}
	return a.complexMap(cmplx.Conj)
}

// Angle returns the argument of each element in radians, as numpy.angle: a
// float array (float32 for complex64, float64 otherwise).
func (a *Array) Angle() *Array {
	if !a.dtype.IsComplex() {
		return a.AsType(Complex128).Angle()
	}
	r := a.complexMap(func(z complex128) complex128 { return complex(cmplx.Phase(z), 0) })
	return r.AsType(complexPart(a.dtype))
}

// complexMap applies f to the elements of a complex array.
func (a *Array) complexMap(f func(complex128) complex128) *Array {
	d := makeStore(a.dtype, a.Size())
	switch s := a.contiguousStore().(type) {
	case []complex64:
		unaryComplex(f, d.([]complex64), s)
	case []complex128:
		unaryComplex(f, d.([]complex128), s)
	}
	return a.result(d, append([]int(nil), a.shape...))
}

// complexPart is the float dtype of one component of a complex dtype.
func complexPart(dt DType) DType {
	if dt == Complex64 {
		return Float32
	}
	return Float64
}
