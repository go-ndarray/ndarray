package ndarray

import "fmt"

// DType is the element type of an Array. It is a runtime property, as in
// NumPy: one Array type holds any of them, and operations on two arrays pick
// the result type from both operands (see ResultType). The zero value is
// Float64, so every array built before dtypes existed keeps its meaning.
type DType uint8

// The element types. Their names, sizes and promotion follow NumPy.
const (
	Float64 DType = iota
	Bool
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float32
	Complex64
	Complex128

	numDTypes = iota
)

var dtypeNames = [numDTypes]string{
	Float64: "float64", Bool: "bool",
	Int8: "int8", Int16: "int16", Int32: "int32", Int64: "int64",
	Uint8: "uint8", Uint16: "uint16", Uint32: "uint32", Uint64: "uint64",
	Float32: "float32", Complex64: "complex64", Complex128: "complex128",
}

// String returns the NumPy name of the dtype ("float64", "int8", ...).
func (d DType) String() string {
	if int(d) < numDTypes {
		return dtypeNames[d]
	}
	return fmt.Sprintf("DType(%d)", uint8(d))
}

// ParseDType returns the dtype with the given NumPy name.
func ParseDType(name string) (DType, error) {
	for d, n := range dtypeNames {
		if n == name {
			return DType(d), nil
		}
	}
	return 0, fmt.Errorf("%w: unknown dtype %q", ErrDType, name)
}

// ErrDType is returned when an operation is not defined for an array's dtype,
// or a dtype name is not known.
var ErrDType = fmt.Errorf("ndarray: unsupported dtype")

// Size returns the size of one element in bytes.
func (d DType) Size() int {
	switch d {
	case Bool, Int8, Uint8:
		return 1
	case Int16, Uint16:
		return 2
	case Int32, Uint32, Float32:
		return 4
	case Complex128:
		return 16
	}
	return 8 // Float64, Int64, Uint64, Complex64
}

// IsBool reports whether d is Bool.
func (d DType) IsBool() bool { return d == Bool }

// IsInteger reports whether d is a signed or unsigned integer type.
func (d DType) IsInteger() bool { return d >= Int8 && d <= Uint64 }

// IsSigned reports whether d is a signed integer type.
func (d DType) IsSigned() bool { return d >= Int8 && d <= Int64 }

// IsFloat reports whether d is Float32 or Float64.
func (d DType) IsFloat() bool { return d == Float32 || d == Float64 }

// IsComplex reports whether d is Complex64 or Complex128.
func (d DType) IsComplex() bool { return d == Complex64 || d == Complex128 }

// signedOf returns the signed integer dtype of the given size in bytes.
// It is asked only for 2, 4 and 8: the type holding an unsigned integer of
// half that size.
func signedOf(size int) DType {
	switch size {
	case 2:
		return Int16
	case 4:
		return Int32
	}
	return Int64
}

// ResultType returns the dtype NumPy gives the result of combining arrays of
// dtypes a and b, numpy.result_type(a, b):
//
//   - Bool gives way to anything.
//   - Two integers of the same signedness give the larger; a signed and an
//     unsigned give the smallest signed type holding both, and Float64 when
//     there is none (Int64 with Uint64).
//   - An integer with a float or complex type gives that type when the
//     integer fits its mantissa exactly (8- and 16-bit integers with Float32
//     and Complex64), else the 64-bit float or complex type.
//   - Floats and complexes give the wider, complex when either is.
//
// The table this produces is checked against NumPy 2.5.3 cell by cell.
func ResultType(a, b DType) DType {
	if a == b {
		return a
	}
	switch {
	case a == Bool:
		return b
	case b == Bool:
		return a
	}
	if a.IsInteger() && b.IsInteger() {
		if a.IsSigned() == b.IsSigned() {
			return maxSize2(a, b)
		}
		s, u := a, b
		if u.IsSigned() {
			s, u = u, s
		}
		if s.Size() > u.Size() {
			return s
		}
		if u.Size() < 8 {
			return signedOf(2 * u.Size())
		}
		return Float64
	}
	// At least one inexact type. An integer becomes the float that holds it.
	a, b = inexactFor(a, b), inexactFor(b, a)
	complexOut := a.IsComplex() || b.IsComplex()
	prec := max(precision(a), precision(b))
	switch {
	case complexOut && prec <= 4:
		return Complex64
	case complexOut:
		return Complex128
	case prec <= 4:
		return Float32
	}
	return Float64
}

// inexactFor maps an integer d, combined with the inexact type other, to the
// float type that represents it: Float32 for 8- and 16-bit integers next to a
// single-precision type, else Float64. Inexact types are returned unchanged.
func inexactFor(d, other DType) DType {
	if !d.IsInteger() {
		return d
	}
	if d.Size() <= 2 && precision(other) <= 4 {
		return Float32
	}
	return Float64
}

// precision is the size in bytes of one real component of an inexact dtype.
func precision(d DType) int {
	if d.IsComplex() {
		return d.Size() / 2
	}
	return d.Size()
}

// maxSize2 returns the larger of two integer dtypes of the same signedness.
func maxSize2(a, b DType) DType {
	if a.Size() >= b.Size() {
		return a
	}
	return b
}

// scalarKind is the kind of a Go scalar operand: under NumPy 2's rules
// (NEP 50) a Python int, float or complex is "weak" — it takes the array's
// dtype when the array's kind can hold it, so float32_array + 1.5 stays
// float32. Go scalars passed to this package behave the same way.
type scalarKind uint8

const (
	strong scalarKind = iota // not a scalar: an ordinary array
	weakInt
	weakFloat
	weakComplex
)

// weakResult returns the dtype of an operation between an array of dtype d
// and a weak scalar of kind k.
func weakResult(d DType, k scalarKind) DType {
	switch k {
	case weakInt:
		if d == Bool {
			return Int64
		}
	case weakFloat:
		if d == Bool || d.IsInteger() {
			return Float64
		}
	case weakComplex:
		switch {
		case d == Float32 || d == Complex64:
			return Complex64
		case !d.IsComplex():
			return Complex128
		}
	}
	return d
}

// promote returns the result dtype of combining a and b, honouring a weak
// scalar operand on either side.
func promote(a, b *Array) DType {
	switch {
	case a.weak != strong && b.weak != strong:
		return weakResult(weakResult(Bool, a.weak), b.weak)
	case b.weak != strong:
		return weakResult(a.dtype, b.weak)
	case a.weak != strong:
		return weakResult(b.dtype, a.weak)
	}
	return ResultType(a.dtype, b.dtype)
}
