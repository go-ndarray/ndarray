package ndarray

import (
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// FuzzMixedDTypes drives the binary operations with operands of two
// arbitrary dtypes, arbitrary shapes (broadcastable or not) and strided views,
// from arbitrary bytes. values.json pins every operation on operands of ONE
// dtype; nothing else checked the values of a mixed-dtype operation.
//
// The oracle is written here, independently of the library:
//   - the result dtype is NumPy's own (testdata/promotion.json, numpy 2.5.3);
//   - the shape is the broadcasting rule;
//   - each element is computed by casting both operands to the result dtype
//     and applying Go's arithmetic of that type, which wraps integers as C and
//     NumPy do; Bool + is "or", Bool * is "and", and Bool - Bool is refused,
//     as NumPy refuses it.
//
// The library must never panic, must refuse exactly what the oracle refuses,
// and must agree on dtype, shape and every element. The seed corpus runs as an
// ordinary test; `go test -fuzz FuzzMixedDTypes` explores.
func FuzzMixedDTypes(f *testing.F) {
	var p promotionTable
	if err := json.Unmarshal(promotionJSON, &p); err != nil {
		f.Fatal(err)
	}
	for _, s := range [][]byte{
		{0, 1, 0, 1, 3, 0, 0, 2, 3, 1, 2},
		{1, 8, 2, 2, 3, 1, 3, 0, 1, 2, 3, 4, 5, 6},
		{2, 5, 9, 1, 1, 2, 4, 0, 7, 7, 7, 7},
		{3, 12, 11, 2, 2, 2, 2, 2, 1, 0, 3, 5},
		{4, 0, 0, 1, 3, 1, 3, 0, 1, 1},
		{5, 6, 10, 2, 3, 1, 2, 3, 0, 4, 2, 6, 1},
		{0, 4, 9, 1, 4, 1, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8},
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		r := &byteReader{b: data}
		op := mixedOps[r.n(len(mixedOps))]
		da, db := DType(r.n(numDTypes)), DType(r.n(numDTypes))
		a := mixedOperand(r, da)
		b := mixedOperand(r, db)
		if a == nil || b == nil {
			return
		}
		var got *Array
		var err error
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("%s(%v %v, %v %v) panicked: %v", op.name, da, a.Shape(), db, b.Shape(), p)
				}
			}()
			got, err = op.call(a, b)
		}()
		shape, ok := broadcastOracle(a.Shape(), b.Shape())
		refused := !ok || (op.name == "Sub" && da == Bool && db == Bool)
		if refused != (err != nil) {
			t.Fatalf("%s(%v %v, %v %v): err = %v, oracle refuses = %v", op.name, da, a.Shape(), db, b.Shape(), err, refused)
		}
		if refused {
			return
		}
		rt, err2 := ParseDType(p.Pair[da.String()+","+db.String()])
		if err2 != nil {
			t.Fatal(err2)
		}
		want := rt
		if op.name == "Equal" {
			want = Bool
		}
		if got.DType() != want || !sameShape(got.Shape(), shape) {
			t.Fatalf("%s(%v %v, %v %v) = %v %v, oracle %v %v", op.name, da, a.Shape(), db, b.Shape(), got.DType(), got.Shape(), want, shape)
		}
		if (op.name == "Maximum" || op.name == "Minimum") && rt.IsComplex() {
			return // NumPy orders complex numbers lexicographically, with its own NaN rules: dtype and shape only
		}
		gs := got.contiguousStore()
		eachIndex(shape, func(idx []int) {
			x := castVal(readAt(a, broadcastIndex(idx, a.Shape())), rt)
			y := castVal(readAt(b, broadcastIndex(idx, b.Shape())), rt)
			w := op.oracle(x, y, rt)
			g := readFlat(gs, flatIndex(idx, shape), got.DType())
			if !sameVal(g, w) {
				t.Fatalf("%s(%v %v, %v %v) at %v = %v, oracle %v (operands %v, %v)",
					op.name, da, a.Shape(), db, b.Shape(), idx, g, w, x, y)
			}
		})
	})
}

type mixedOp struct {
	name   string
	call   func(a, b *Array) (*Array, error)
	oracle func(x, y val, rt DType) val
}

var mixedOps = []mixedOp{
	{"Add", (*Array).Add, func(x, y val, rt DType) val { return arith(x, y, rt, '+') }},
	{"Sub", (*Array).Sub, func(x, y val, rt DType) val { return arith(x, y, rt, '-') }},
	{"Mul", (*Array).Mul, func(x, y val, rt DType) val { return arith(x, y, rt, '*') }},
	{"Maximum", (*Array).Maximum, func(x, y val, rt DType) val { return extremum(x, y, rt, true) }},
	{"Minimum", (*Array).Minimum, func(x, y val, rt DType) val { return extremum(x, y, rt, false) }},
	{"Equal", (*Array).Equal, func(x, y val, rt DType) val { return val{dt: Bool, b: equalVal(x, y)} }},
}

// val is one element, in the field its dtype uses.
type val struct {
	dt DType
	b  bool
	i  int64
	u  uint64
	f  float64
	c  complex128
}

func (v val) String() string {
	switch {
	case v.dt == Bool:
		return fmt.Sprint(v.b)
	case v.dt.IsSigned():
		return fmt.Sprintf("%d %v", v.i, v.dt)
	case v.dt.IsInteger():
		return fmt.Sprintf("%d %v", v.u, v.dt)
	case v.dt.IsComplex():
		return fmt.Sprintf("%v %v", v.c, v.dt)
	}
	return fmt.Sprintf("%v %v", v.f, v.dt)
}

// edgeValues are the values an operand of each kind is filled from.
var (
	edgeInts   = []int64{0, 1, -1, 2, -3, 7, math.MaxInt64, math.MinInt64, math.MaxInt32, math.MinInt32, 127, -128, 255, 32767, -32768, 65535}
	edgeUints  = []uint64{0, 1, 2, 3, 7, 127, 128, 255, 256, 65535, math.MaxUint32, math.MaxUint64, 1 << 63}
	edgeFloats = []float64{0, math.Copysign(0, -1), 1, -1, 2.5, -0.75, 1e30, -1e-30, math.MaxFloat64, math.SmallestNonzeroFloat64, math.Inf(1), math.Inf(-1), math.NaN(), 3}
)

// mixedOperand builds an array of dtype dt, of 0 to 3 axes of length 0 to 3,
// sometimes transposed (a strided view).
func mixedOperand(r *byteReader, dt DType) *Array {
	shape := make([]int, r.n(4))
	for i := range shape {
		shape[i] = r.n(4)
	}
	n := prod(shape)
	vals := make([]val, n)
	for i := range vals {
		k := int(r.next())
		switch {
		case dt == Bool:
			vals[i] = val{dt: Bool, b: k%2 == 1}
		case dt.IsSigned():
			vals[i] = castVal(val{dt: Int64, i: edgeInts[k%len(edgeInts)]}, dt)
		case dt.IsInteger():
			vals[i] = castVal(val{dt: Uint64, u: edgeUints[k%len(edgeUints)]}, dt)
		case dt.IsComplex():
			re, im := edgeFloats[k%len(edgeFloats)], edgeFloats[(k/len(edgeFloats))%len(edgeFloats)]
			vals[i] = castVal(val{dt: Complex128, c: complex(re, im)}, dt)
		default:
			vals[i] = castVal(val{dt: Float64, f: edgeFloats[k%len(edgeFloats)]}, dt)
		}
	}
	s := makeStore(dt, n)
	for i, v := range vals {
		writeFlat(s, i, v)
	}
	a := fromStore(s, shape)
	if len(shape) >= 2 && r.n(2) == 1 {
		a = a.Transpose()
	}
	return a
}

// castVal converts v to dt as a C cast does for the conversions promotion
// makes (always to a dtype that holds the kind of v: bool to anything,
// integers to wider integers or to floats, reals to complex).
func castVal(v val, dt DType) val {
	out := val{dt: dt}
	var asI int64
	var asU uint64
	var asF float64
	var asC complex128
	switch {
	case v.dt == Bool:
		if v.b {
			asI, asU, asF, asC = 1, 1, 1, 1
		}
	case v.dt.IsSigned():
		asI, asU, asF, asC = v.i, uint64(v.i), float64(v.i), complex(float64(v.i), 0)
	case v.dt.IsInteger():
		asI, asU, asF, asC = int64(v.u), v.u, float64(v.u), complex(float64(v.u), 0)
	case v.dt.IsComplex():
		asF, asC = real(v.c), v.c
	default:
		asF, asC = v.f, complex(v.f, 0)
	}
	switch dt {
	case Bool:
		out.b = asC != 0
	case Int8:
		out.i = int64(int8(asI))
	case Int16:
		out.i = int64(int16(asI))
	case Int32:
		out.i = int64(int32(asI))
	case Int64:
		out.i = asI
	case Uint8:
		out.u = uint64(uint8(asU))
	case Uint16:
		out.u = uint64(uint16(asU))
	case Uint32:
		out.u = uint64(uint32(asU))
	case Uint64:
		out.u = asU
	case Float32:
		out.f = float64(float32(asF))
	case Float64:
		out.f = asF
	case Complex64:
		out.c = complex128(complex64(asC))
	case Complex128:
		out.c = asC
	}
	return out
}

// arith applies + - * in dtype rt, with that type's own arithmetic.
func arith(x, y val, rt DType, op byte) val {
	out := val{dt: rt}
	switch rt {
	case Bool:
		if op == '+' {
			out.b = x.b || y.b
		} else {
			out.b = x.b && y.b
		}
	case Int8:
		out.i = int64(intOp(int8(x.i), int8(y.i), op))
	case Int16:
		out.i = int64(intOp(int16(x.i), int16(y.i), op))
	case Int32:
		out.i = int64(intOp(int32(x.i), int32(y.i), op))
	case Int64:
		out.i = intOp(x.i, y.i, op)
	case Uint8:
		out.u = uint64(intOp(uint8(x.u), uint8(y.u), op))
	case Uint16:
		out.u = uint64(intOp(uint16(x.u), uint16(y.u), op))
	case Uint32:
		out.u = uint64(intOp(uint32(x.u), uint32(y.u), op))
	case Uint64:
		out.u = intOp(x.u, y.u, op)
	case Float32:
		out.f = float64(intOp(float32(x.f), float32(y.f), op))
	case Float64:
		out.f = intOp(x.f, y.f, op)
	case Complex64:
		out.c = complex128(intOp(complex64(x.c), complex64(y.c), op))
	case Complex128:
		out.c = intOp(x.c, y.c, op)
	}
	return out
}

func intOp[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64 | complex64 | complex128](a, b T, op byte) T {
	switch op {
	case '+':
		return a + b
	case '-':
		return a - b
	}
	return a * b
}

// extremum is NumPy's maximum/minimum on real dtypes: a NaN operand wins.
func extremum(x, y val, rt DType, isMax bool) val {
	out := val{dt: rt}
	switch {
	case rt == Bool:
		out.b = x.b || y.b
		if !isMax {
			out.b = x.b && y.b
		}
	case rt.IsSigned():
		out.i = max(x.i, y.i)
		if !isMax {
			out.i = min(x.i, y.i)
		}
	case rt.IsInteger():
		out.u = max(x.u, y.u)
		if !isMax {
			out.u = min(x.u, y.u)
		}
	default:
		switch {
		case math.IsNaN(x.f):
			out.f = x.f
		case math.IsNaN(y.f):
			out.f = y.f
		case isMax:
			out.f = math.Max(x.f, y.f)
		default:
			out.f = math.Min(x.f, y.f)
		}
	}
	return out
}

func equalVal(x, y val) bool {
	switch {
	case x.dt == Bool:
		return x.b == y.b
	case x.dt.IsSigned():
		return x.i == y.i
	case x.dt.IsInteger():
		return x.u == y.u
	case x.dt.IsComplex():
		return x.c == y.c
	}
	return x.f == y.f
}

// sameVal compares a result with the oracle: exactly, NaN equal to NaN of
// any payload, and complex products within one part in 2^-20 of the largest
// part (Go may fuse a*b+c into one rounding on arm64, ppc64le and s390x).
func sameVal(g, w val) bool {
	if g.dt != w.dt {
		return false
	}
	switch {
	case g.dt.IsComplex():
		if cmplxIsNaN(g.c) || cmplxIsNaN(w.c) {
			return cmplxIsNaN(g.c) && cmplxIsNaN(w.c)
		}
		if g.c == w.c {
			return true
		}
		if cmplx.IsInf(g.c) || cmplx.IsInf(w.c) {
			return false
		}
		scale := max(math.Abs(real(w.c)), math.Abs(imag(w.c)), math.SmallestNonzeroFloat64)
		return cabs(g.c-w.c) <= scale*0x1p-20
	case g.dt == Float32 || g.dt == Float64:
		if math.IsNaN(g.f) || math.IsNaN(w.f) {
			return math.IsNaN(g.f) && math.IsNaN(w.f)
		}
		return g.f == w.f && math.Signbit(g.f) == math.Signbit(w.f)
	}
	return equalVal(g, w)
}

// readAt reads the element of a at index idx (any layout).
func readAt(a *Array, idx []int) val {
	s := a.contiguousStore()
	return readFlat(s, flatIndex(idx, a.Shape()), a.DType())
}

func readFlat(s any, i int, dt DType) val {
	v := val{dt: dt}
	switch st := s.(type) {
	case []bool:
		v.b = st[i]
	case []int8:
		v.i = int64(st[i])
	case []int16:
		v.i = int64(st[i])
	case []int32:
		v.i = int64(st[i])
	case []int64:
		v.i = st[i]
	case []uint8:
		v.u = uint64(st[i])
	case []uint16:
		v.u = uint64(st[i])
	case []uint32:
		v.u = uint64(st[i])
	case []uint64:
		v.u = st[i]
	case []float32:
		v.f = float64(st[i])
	case []float64:
		v.f = st[i]
	case []complex64:
		v.c = complex128(st[i])
	case []complex128:
		v.c = st[i]
	}
	return v
}

func writeFlat(s any, i int, v val) {
	switch st := s.(type) {
	case []bool:
		st[i] = v.b
	case []int8:
		st[i] = int8(v.i)
	case []int16:
		st[i] = int16(v.i)
	case []int32:
		st[i] = int32(v.i)
	case []int64:
		st[i] = v.i
	case []uint8:
		st[i] = uint8(v.u)
	case []uint16:
		st[i] = uint16(v.u)
	case []uint32:
		st[i] = uint32(v.u)
	case []uint64:
		st[i] = v.u
	case []float32:
		st[i] = float32(v.f)
	case []float64:
		st[i] = v.f
	case []complex64:
		st[i] = complex64(v.c)
	case []complex128:
		st[i] = v.c
	}
}

// broadcastOracle is NumPy's broadcasting rule.
func broadcastOracle(a, b []int) ([]int, bool) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 1; i <= n; i++ {
		x, y := 1, 1
		if i <= len(a) {
			x = a[len(a)-i]
		}
		if i <= len(b) {
			y = b[len(b)-i]
		}
		switch {
		case x == y || y == 1:
			out[n-i] = x
		case x == 1:
			out[n-i] = y
		default:
			return nil, false
		}
	}
	return out, true
}

// broadcastIndex maps an index of the broadcast result to the operand's.
func broadcastIndex(idx, shape []int) []int {
	out := make([]int, len(shape))
	off := len(idx) - len(shape)
	for i := range shape {
		if shape[i] != 1 {
			out[i] = idx[off+i]
		}
	}
	return out
}

func flatIndex(idx, shape []int) int {
	f := 0
	for i := range shape {
		f = f*shape[i] + idx[i]
	}
	return f
}
