package ndarray

import (
	"math"
	"math/cmplx"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// Inner loops for the dtypes other than Float64. Each takes contiguous storage
// slices of one element type, already converted to the result dtype and
// broadcast to the result shape; the operation code is switched on once, so
// every loop is a tight loop over one element type. The generated
// zz_dtypes_gen.go picks the instantiation from the storage type.

// binCode selects a binary elementwise operation.
type binCode uint8

const (
	opAdd binCode = iota
	opSub
	opMul
	opDiv
	opMax
	opMin
)

// cmpCode selects a comparison.
type cmpCode uint8

const (
	opEq cmpCode = iota
	opNe
	opLt
	opLe
	opGt
	opGe
)

// binNum applies an arithmetic operation. opDiv is only used for inexact
// types: integer division is true division, done in a float type.
func binNum[T numT](op binCode, d, x, y []T) {
	switch op {
	case opAdd:
		for i := range d {
			d[i] = x[i] + y[i]
		}
	case opSub:
		for i := range d {
			d[i] = x[i] - y[i]
		}
	case opMul:
		for i := range d {
			d[i] = x[i] * y[i]
		}
	case opDiv:
		for i := range d {
			d[i] = x[i] / y[i]
		}
	}
}

// binInt applies an arithmetic or max/min operation to integers.
func binInt[T intT](op binCode, d, x, y []T) {
	switch op {
	case opMax:
		for i := range d {
			d[i] = max(x[i], y[i])
		}
	case opMin:
		for i := range d {
			d[i] = min(x[i], y[i])
		}
	default:
		binNum(op, d, x, y)
	}
}

// binFloat applies an arithmetic or max/min operation to floats. Max and min
// propagate NaN, as NumPy's maximum and minimum do (and Go's builtins).
func binFloat[T floatT](op binCode, d, x, y []T) {
	switch op {
	case opMax:
		for i := range d {
			d[i] = max(x[i], y[i])
		}
	case opMin:
		for i := range d {
			d[i] = min(x[i], y[i])
		}
	default:
		binNum(op, d, x, y)
	}
}

// binComplex applies an arithmetic or max/min operation to complex numbers.
// Max and min order lexicographically, real part first, and propagate NaN, as
// NumPy does.
func binComplex[T complexT](op binCode, d, x, y []T) {
	switch op {
	case opMax:
		for i := range d {
			d[i] = pickComplex(x[i], y[i], true)
		}
	case opMin:
		for i := range d {
			d[i] = pickComplex(x[i], y[i], false)
		}
	default:
		binNum(op, d, x, y)
	}
}

// pickComplex returns the larger (or smaller) of x and y in NumPy's
// lexicographic order; a NaN in either gives that operand, x first.
func pickComplex[T complexT](x, y T, larger bool) T {
	xc, yc := complex128(x), complex128(y)
	if cmplx.IsNaN(xc) {
		return x
	}
	if cmplx.IsNaN(yc) {
		return y
	}
	if lessComplex(xc, yc) == larger {
		return y
	}
	return x
}

// lessComplex is NumPy's lexicographic order on complex numbers.
func lessComplex(x, y complex128) bool {
	return real(x) < real(y) || (real(x) == real(y) && imag(x) < imag(y))
}

// binBool applies an operation to booleans: NumPy's add and maximum are "or",
// multiply and minimum are "and". Subtraction is an error before this is
// reached, and division converts to a float type first.
func binBool(op binCode, d, x, y []bool) {
	switch op {
	case opAdd, opMax:
		for i := range d {
			d[i] = x[i] || y[i]
		}
	case opMul, opMin:
		for i := range d {
			d[i] = x[i] && y[i]
		}
	}
}

// cmpOrdered compares real values.
func cmpOrdered[T realT](op cmpCode, d []bool, x, y []T) {
	switch op {
	case opEq:
		for i := range d {
			d[i] = x[i] == y[i]
		}
	case opNe:
		for i := range d {
			d[i] = x[i] != y[i]
		}
	case opLt:
		for i := range d {
			d[i] = x[i] < y[i]
		}
	case opLe:
		for i := range d {
			d[i] = x[i] <= y[i]
		}
	case opGt:
		for i := range d {
			d[i] = x[i] > y[i]
		}
	case opGe:
		for i := range d {
			d[i] = x[i] >= y[i]
		}
	}
}

// cmpComplex compares complex values: equality componentwise, order
// lexicographic as NumPy defines it (false when either side has a NaN).
func cmpComplex[T complexT](op cmpCode, d []bool, x, y []T) {
	for i := range d {
		a, b := complex128(x[i]), complex128(y[i])
		nan := cmplx.IsNaN(a) || cmplx.IsNaN(b)
		switch op {
		case opEq:
			d[i] = a == b
		case opNe:
			d[i] = a != b
		case opLt:
			d[i] = !nan && lessComplex(a, b)
		case opLe:
			d[i] = !nan && !lessComplex(b, a)
		case opGt:
			d[i] = !nan && lessComplex(b, a)
		case opGe:
			d[i] = !nan && !lessComplex(a, b)
		}
	}
}

// cmpBool compares booleans, with false < true.
func cmpBool(op cmpCode, d []bool, x, y []bool) {
	for i := range d {
		a, b := x[i], y[i]
		switch op {
		case opEq:
			d[i] = a == b
		case opNe:
			d[i] = a != b
		case opLt:
			d[i] = !a && b
		case opLe:
			d[i] = !a || b
		case opGt:
			d[i] = a && !b
		case opGe:
			d[i] = a || !b
		}
	}
}

// unaryCode selects a unary operation on an inexact type.
type unaryCode uint8

const (
	uSqrt unaryCode = iota
	uExp
	uLog
	uLog2
	uLog10
	uSin
	uCos
	uTan
	uFloor
	uCeil
	uRound
	uSquare
	uNeg
	uAbs
	uPow
)

// realFuncs are the float64 functions behind the unary codes.
var realFuncs = [...]func(float64) float64{
	uSqrt: math.Sqrt, uExp: math.Exp, uLog: math.Log, uLog2: log2,
	uLog10: math.Log10, uSin: math.Sin, uCos: math.Cos, uTan: math.Tan,
	uFloor: math.Floor, uCeil: math.Ceil, uRound: math.Round,
	uSquare: func(x float64) float64 { return x * x },
	uNeg:    func(x float64) float64 { return -x },
	uAbs:    math.Abs,
}

// complexFuncs are the complex128 functions behind the unary codes; nil where
// NumPy does not define the operation on complex numbers.
var complexFuncs = [...]func(complex128) complex128{
	uSqrt: cmplx.Sqrt, uExp: cmplx.Exp, uLog: cmplx.Log,
	uLog2:  func(z complex128) complex128 { return cmplx.Log(z) / math.Ln2 },
	uLog10: cmplx.Log10, uSin: cmplx.Sin, uCos: cmplx.Cos, uTan: cmplx.Tan,
	uFloor: nil, uCeil: nil,
	uRound:  func(z complex128) complex128 { return complex(math.Round(real(z)), math.Round(imag(z))) },
	uSquare: func(z complex128) complex128 { return z * z },
	uNeg:    func(z complex128) complex128 { return -z },
	uAbs:    nil, // complex abs is real-valued: see absStore
}

// unaryFloat32 applies f, computed in float64 and rounded, to float32
// elements. For the correctly rounded operations (sqrt, floor, ceil, round,
// negation, abs, square) this is exactly the float32 result: float64 carries
// more than twice float32's precision, so the second rounding cannot change it.
func unaryFloat32(f func(float64) float64, d, x []float32) {
	for i := range d {
		d[i] = float32(f(float64(x[i])))
	}
}

// unaryComplex applies f to complex elements, computing complex64 in
// complex128.
func unaryComplex[T complexT](f func(complex128) complex128, d, x []T) {
	for i := range d {
		d[i] = T(f(complex128(x[i])))
	}
}

// powFloat32 raises float32 elements to p, computed in float64.
func powFloat32(p float64, d, x []float32) {
	for i := range d {
		d[i] = float32(math.Pow(float64(x[i]), p))
	}
}

// powComplex raises complex elements to p.
func powComplex[T complexT](p complex128, d, x []T) {
	for i := range d {
		d[i] = T(cmplx.Pow(complex128(x[i]), p))
	}
}

// negInt negates integers, wrapping as Go and NumPy do.
func negInt[T intT](d, x []T) {
	for i := range d {
		d[i] = -x[i]
	}
}

// absInt is the absolute value of integers; the minimum of a signed type is
// its own absolute value, as in NumPy.
func absInt[T intT](d, x []T) {
	for i := range d {
		v := x[i]
		if v < 0 {
			v = -v
		}
		d[i] = v
	}
}

// squareInt squares integers, wrapping.
func squareInt[T intT](d, x []T) {
	for i := range d {
		d[i] = x[i] * x[i]
	}
}

// absComplex is the modulus of complex elements, into the real type of the
// same precision.
func absComplex[T complexT, R floatT](d []R, x []T) {
	for i := range d {
		d[i] = R(cmplx.Abs(complex128(x[i])))
	}
}

// Reductions over the [outer][axisLen][inner] layout the axis reductions use:
// dst[o*inner+j] combines src[(o*axisLen+k)*inner+j] over k. axisLen > 0.

// reduceCode selects a reduction.
type reduceCode uint8

const (
	rSum reduceCode = iota
	rProd
	rMax
	rMin
)

// reduceNum sums or multiplies along the axis.
func reduceNum[T numT](op reduceCode, dst, src []T, outer, axisLen, inner int) {
	if inner == 1 {
		// The reduced axis is contiguous (the last axis, or the whole
		// array): fold each run directly. The loop below would take a
		// slice and run a one-iteration loop per element there, 6.8 ns an
		// element for a float32 sum on POWER9.
		for o := 0; o < outer; o++ {
			dst[o] = foldRun(op, src[o*axisLen:(o+1)*axisLen])
		}
		return
	}
	for o := 0; o < outer; o++ {
		row := dst[o*inner : (o+1)*inner]
		copy(row, src[o*axisLen*inner:(o*axisLen+1)*inner])
		for k := 1; k < axisLen; k++ {
			s := src[(o*axisLen+k)*inner : (o*axisLen+k+1)*inner]
			if op == rSum {
				for j := range row {
					row[j] += s[j]
				}
			} else {
				for j := range row {
					row[j] *= s[j]
				}
			}
		}
	}
}

// foldRun reduces one contiguous run (len(s) > 0) as NumPy's add and
// multiply reductions do: the result starts as the first element and the
// rest is added to it, pairwise for a floating sum (pairwiseSum), in order
// otherwise. Integer sums are exact in any order, and NumPy multiplies in
// order.
func foldRun[T numT](op reduceCode, s []T) T {
	acc := s[0]
	rest := s[1:]
	if op == rSum {
		switch r := any(rest).(type) {
		case []float32: // Float64 sums never come here: they have kernels
			return any(any(acc).(float32) + pairwiseSum(r)).(T)
		case []complex64:
			return any(any(acc).(complex64) + pairwiseSumComplex(r)).(T)
		case []complex128:
			return any(any(acc).(complex128) + pairwiseSumComplex(r)).(T)
		}
		for _, v := range rest {
			acc += v
		}
		return acc
	}
	for _, v := range rest {
		acc *= v
	}
	return acc
}

// foldWidened sums or multiplies each contiguous run of src (bools or
// integers narrower than 64 bits) into dst, NumPy's accumulator for them:
// []int64 for bools and signed integers, []uint64 for unsigned. Each element
// is widened as it is read; the arithmetic, wrap-around included, is that of
// converting first and folding after.
func foldWidened(op reduceCode, dst, src any, outer, axisLen int) {
	switch s := src.(type) {
	case []bool:
		d := dst.([]int64)
		for o := range outer {
			var acc int64 = 1 // a product of bools is 1 unless one is false
			if op == rSum {
				acc = 0
			}
			for _, v := range s[o*axisLen : (o+1)*axisLen] {
				switch {
				case op == rSum && v:
					acc++
				case op == rProd && !v:
					acc = 0
				}
			}
			d[o] = acc
		}
	case []int8:
		widenRuns(op, dst.([]int64), s, outer, axisLen)
	case []int16:
		widenRuns(op, dst.([]int64), s, outer, axisLen)
	case []int32:
		widenRuns(op, dst.([]int64), s, outer, axisLen)
	case []uint8:
		widenRuns(op, dst.([]uint64), s, outer, axisLen)
	case []uint16:
		widenRuns(op, dst.([]uint64), s, outer, axisLen)
	case []uint32:
		widenRuns(op, dst.([]uint64), s, outer, axisLen)
	}
}

func widenRuns[S int8 | int16 | int32 | uint8 | uint16 | uint32, D int64 | uint64](op reduceCode, dst []D, src []S, outer, axisLen int) {
	for o := range outer {
		run := src[o*axisLen : (o+1)*axisLen]
		if op == rSum {
			dst[o] = sumWiden[S, D](run)
		} else {
			dst[o] = prodWiden[S, D](run)
		}
	}
}

// sumWiden and prodWiden fold a run with four accumulators. Integer addition
// and multiplication wrap modulo 2^64, so they are associative and the order
// changes nothing: the result is the in-order fold's, exactly. One loop per
// operation, so the operation is not re-read per element (on amd64 the
// single loop reloaded it from the stack every iteration), and four
// independent chains, so the loop is no longer one dependent add an element:
// it had run 50% slower on a Zen 3 just by being moved across a 64-byte line.
func sumWiden[S int8 | int16 | int32 | uint8 | uint16 | uint32, D int64 | uint64](s []S) D {
	var a0, a1, a2, a3 D
	i := 0
	for ; i+4 <= len(s); i += 4 {
		a0 += D(s[i])
		a1 += D(s[i+1])
		a2 += D(s[i+2])
		a3 += D(s[i+3])
	}
	for ; i < len(s); i++ {
		a0 += D(s[i])
	}
	return (a0 + a1) + (a2 + a3)
}

func prodWiden[S int8 | int16 | int32 | uint8 | uint16 | uint32, D int64 | uint64](s []S) D {
	var a0, a1, a2, a3 D = 1, 1, 1, 1
	i := 0
	for ; i+4 <= len(s); i += 4 {
		a0 *= D(s[i])
		a1 *= D(s[i+1])
		a2 *= D(s[i+2])
		a3 *= D(s[i+3])
	}
	for ; i < len(s); i++ {
		a0 *= D(s[i])
	}
	return (a0 * a1) * (a2 * a3)
}

// pwBlock is NumPy's PW_BLOCKSIZE: runs up to this length are summed with
// eight accumulators, longer ones are halved.
const pwBlock = 128

// pairwiseSum is NumPy's FLOAT_pairwise_sum
// (numpy/_core/src/umath/loops_utils.h.src, v2.5.3): rounding error O(lg n)
// instead of O(n), in the same order of additions.
//
// The eight accumulators are fed by kernels.AddBlocks8F32, a vector kernel on
// amd64 and arm64 that adds the same lanes in the same order, so the sums are
// the scalar loop's bit for bit.
func pairwiseSum(a []float32) float32 {
	n := len(a)
	switch {
	case n < 8:
		// From -0, so that a sum of only -0 stays -0.
		res := float32(math.Copysign(0, -1))
		for _, v := range a {
			res += v
		}
		return res
	case n <= pwBlock:
		var r [8]float32
		copy(r[:], a[:8])
		i := n - n%8
		kernels.AddBlocks8F32(&r, a[8:i])
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return pairwiseSum(a[:n2]) + pairwiseSum(a[n2:])
}

// pairwiseSumComplex is NumPy's CFLOAT/CDOUBLE_pairwise_sum: the same scheme
// over the interleaved real and imaginary parts, counted in floats (two per
// element), so four accumulators per part and blocks of 64 elements.
func pairwiseSumComplex[C complex64 | complex128](a []C) C {
	n := 2 * len(a) // NumPy's n counts floats
	switch {
	case n < 8:
		re, im := math.Copysign(0, -1), math.Copysign(0, -1)
		var acc C = C(complex(re, im))
		for _, v := range a {
			acc += v
		}
		return acc
	case n <= pwBlock:
		// r[0], r[2], r[4], r[6] hold real parts and r[1], r[3], r[5], r[7]
		// imaginary parts, of elements 0..3 of each block of four.
		var r [4]C
		copy(r[:], a[:4])
		i := 4
		for ; i < len(a)-len(a)%4; i += 4 {
			r[0] += a[i]
			r[1] += a[i+1]
			r[2] += a[i+2]
			r[3] += a[i+3]
		}
		res := (r[0] + r[1]) + (r[2] + r[3])
		for ; i < len(a); i++ {
			res += a[i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return pairwiseSumComplex(a[:n2/2]) + pairwiseSumComplex(a[n2/2:])
}

// reduceOrdered takes the maximum or minimum along the axis; for floats a NaN
// wins, as in NumPy.
func reduceOrdered[T realT](op reduceCode, dst, src []T, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		row := dst[o*inner : (o+1)*inner]
		copy(row, src[o*axisLen*inner:(o*axisLen+1)*inner])
		for k := 1; k < axisLen; k++ {
			s := src[(o*axisLen+k)*inner : (o*axisLen+k+1)*inner]
			if op == rMax {
				for j := range row {
					row[j] = max(row[j], s[j])
				}
			} else {
				for j := range row {
					row[j] = min(row[j], s[j])
				}
			}
		}
	}
}

// reduceComplex takes the lexicographic maximum or minimum along the axis.
func reduceComplex[T complexT](op reduceCode, dst, src []T, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		row := dst[o*inner : (o+1)*inner]
		copy(row, src[o*axisLen*inner:(o*axisLen+1)*inner])
		for k := 1; k < axisLen; k++ {
			s := src[(o*axisLen+k)*inner : (o*axisLen+k+1)*inner]
			for j := range row {
				row[j] = pickComplex(row[j], s[j], op == rMax)
			}
		}
	}
}

// reduceBool is "any" for the maximum and "all" for the minimum.
func reduceBool(op reduceCode, dst, src []bool, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		row := dst[o*inner : (o+1)*inner]
		copy(row, src[o*axisLen*inner:(o*axisLen+1)*inner])
		for k := 1; k < axisLen; k++ {
			s := src[(o*axisLen+k)*inner : (o*axisLen+k+1)*inner]
			for j := range row {
				if op == rMax {
					row[j] = row[j] || s[j]
				} else {
					row[j] = row[j] && s[j]
				}
			}
		}
	}
}

// argOrdered writes the index of the first maximum (or minimum) along the
// axis; for floats the first NaN wins, as in NumPy.
func argOrdered[T realT](larger bool, dst []int64, src []T, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		for j := 0; j < inner; j++ {
			best, bi := src[o*axisLen*inner+j], 0
			for k := 1; k < axisLen && best == best; k++ {
				v := src[(o*axisLen+k)*inner+j]
				if v != v || (larger && v > best) || (!larger && v < best) {
					best, bi = v, k
				}
			}
			dst[o*inner+j] = int64(bi)
		}
	}
}

// argComplex is argOrdered in NumPy's lexicographic complex order.
func argComplex[T complexT](larger bool, dst []int64, src []T, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		for j := 0; j < inner; j++ {
			best, bi := complex128(src[o*axisLen*inner+j]), 0
			for k := 1; k < axisLen && !cmplx.IsNaN(best); k++ {
				v := complex128(src[(o*axisLen+k)*inner+j])
				if cmplx.IsNaN(v) || (larger && lessComplex(best, v)) || (!larger && lessComplex(v, best)) {
					best, bi = v, k
				}
			}
			dst[o*inner+j] = int64(bi)
		}
	}
}

// argBool is argOrdered with false < true.
func argBool(larger bool, dst []int64, src []bool, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		for j := 0; j < inner; j++ {
			bi := 0
			for k := 0; k < axisLen; k++ {
				if src[(o*axisLen+k)*inner+j] == larger {
					bi = k
					break
				}
			}
			dst[o*inner+j] = int64(bi)
		}
	}
}

// scanNum is the cumulative sum or product along the axis; dst has src's
// layout.
func scanNum[T numT](op reduceCode, dst, src []T, outer, axisLen, inner int) {
	for o := 0; o < outer; o++ {
		base := o * axisLen * inner
		copy(dst[base:base+inner], src[base:base+inner])
		for k := 1; k < axisLen; k++ {
			prev := dst[base+(k-1)*inner : base+k*inner]
			cur := dst[base+k*inner : base+(k+1)*inner]
			s := src[base+k*inner : base+(k+1)*inner]
			for j := range cur {
				if op == rSum {
					cur[j] = prev[j] + s[j]
				} else {
					cur[j] = prev[j] * s[j]
				}
			}
		}
	}
}

// clipOrdered limits real elements to [lo, hi]; NaN stays NaN, as in NumPy.
func clipOrdered[T realT](d, x []T, lo, hi T) {
	for i := range d {
		d[i] = min(max(x[i], lo), hi)
	}
}

// whereStore picks t where c is true and f otherwise.
func whereT[T elemT](d []T, c []bool, t, f []T) {
	for i := range d {
		if c[i] {
			d[i] = t[i]
		} else {
			d[i] = f[i]
		}
	}
}

// matmulNum multiplies an (m x k) by a (k x n) row-major matrix into d.
func matmulNum[T numT](d, x, y []T, m, k, n int) {
	for i := 0; i < m; i++ {
		row := d[i*n : (i+1)*n]
		for p := 0; p < k; p++ {
			xv := x[i*k+p]
			yr := y[p*n : (p+1)*n]
			for j := range row {
				row[j] += xv * yr[j]
			}
		}
	}
}

// matmulBool is the matrix product over the booleans: "or" of "and"s.
func matmulBool(d, x, y []bool, m, k, n int) {
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			if !x[i*k+p] {
				continue
			}
			for j := 0; j < n; j++ {
				d[i*n+j] = d[i*n+j] || y[p*n+j]
			}
		}
	}
}
