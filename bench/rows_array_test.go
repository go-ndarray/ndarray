package bench

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

// row is one measured case: a Go body, and the NumPy expression it is compared
// with. call is a Python expression whose value is called as call(*args).
type row struct {
	name, fn string
	ref      string
	args     []ds
	call     string
	body     func(a []*ndarray.Array) func()
}

var (
	sinkA  *ndarray.Array
	sinkF  float64
	sinkI  int
	sinkB  bool
	sinkAs []*ndarray.Array
)

func must(a *ndarray.Array, err error) *ndarray.Array {
	if err != nil {
		panic(err)
	}
	return a
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}

const (
	nvec = 100_000 // elements of a vector
	mrow = 1000    // a matrix of 1000 x 100 elements for the axis reductions
	mcol = 100
	mm   = 128 // side of the square matrices of the matrix products
)

var (
	dX  = dvec(0.5, 4, nvec, 0)       // positive, away from 0
	dY  = dvec(1, 2, nvec, 3)         // a divisor
	dP  = dvec(0.999, 1.001, nvec, 5) // a product that stays near 1
	dXM = dreshape(dX, mrow, mcol)
	dPM = dreshape(dP, mrow, mcol)
	dX1 = dvec(0.5, 4, mrow, 0) // 1000 points for Outer
	dY1 = dvec(1, 2, mrow, 3)
	dA  = dmat(0, 1, mm, mm, 0)
	dB  = dmat(0, 1, mm, mm, 11)
)

// maskPy is the mask x > 2 on dX, built in the factory so it is outside the timed call.
var maskPy = vpy(0.5, 4, nvec, 0) + " > 2.0"

func un(name, py string, g func(a *ndarray.Array) *ndarray.Array) row {
	return row{name: "Array." + name, fn: "Array." + name, ref: "numpy " + py, args: []ds{dX},
		call: py, body: func(a []*ndarray.Array) func() { return func() { sinkA = g(a[0]) } }}
}

func bin(name, py string, g func(a, b *ndarray.Array) (*ndarray.Array, error)) row {
	return row{name: "Array." + name, fn: "Array." + name, ref: "numpy " + py, args: []ds{dX, dY},
		call: py, body: func(a []*ndarray.Array) func() { return func() { sinkA = must(g(a[0], a[1])) } }}
}

func into(name, py string, g func(a, out, b *ndarray.Array) error) row {
	return row{name: "Array." + name, fn: "Array." + name, ref: "numpy " + py + " (out= preallocated)", args: []ds{dX, dY},
		call: fmt.Sprintf("(lambda o: lambda x, y: %s(x, y, out=o))(np.empty(%d))", py, nvec),
		body: func(a []*ndarray.Array) func() {
			out := must(ndarray.New(nvec))
			return func() { chk(g(a[0], out, a[1])) }
		}}
}

func scal(name, op string, g func(a *ndarray.Array, v float64) *ndarray.Array) row {
	return row{name: "Array." + name, fn: "Array." + name, ref: "numpy x " + op + " 1.5", args: []ds{dX},
		call: "lambda x: x " + op + " 1.5", body: func(a []*ndarray.Array) func() { return func() { sinkA = g(a[0], 1.5) } }}
}

// axisRows gives a reduction over axis 0 and over axis 1 of a 1000 x 100 matrix.
func axisRows(name, py string, d ds, g func(a *ndarray.Array, axis int, keep bool) (*ndarray.Array, error)) []row {
	var out []row
	for ax := 0; ax < 2; ax++ {
		out = append(out, row{name: fmt.Sprintf("Array.%s/axis%d", name, ax), fn: "Array." + name,
			ref: fmt.Sprintf("numpy %s(axis=%d)", py, ax), args: []ds{d},
			call: fmt.Sprintf("lambda x: %s(x, axis=%d)", py, ax),
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(g(a[0], ax, false)) } }})
	}
	return out
}

func arrayRows() []row {
	rs := []row{
		un("Abs", "np.abs", (*ndarray.Array).Abs),
		un("Neg", "np.negative", (*ndarray.Array).Neg),
		un("Sqrt", "np.sqrt", (*ndarray.Array).Sqrt),
		un("Square", "np.square", (*ndarray.Array).Square),
		un("Exp", "np.exp", (*ndarray.Array).Exp),
		un("Log", "np.log", (*ndarray.Array).Log),
		un("Log2", "np.log2", (*ndarray.Array).Log2),
		un("Log10", "np.log10", (*ndarray.Array).Log10),
		un("Sin", "np.sin", (*ndarray.Array).Sin),
		un("Cos", "np.cos", (*ndarray.Array).Cos),
		un("Tan", "np.tan", (*ndarray.Array).Tan),
		un("Ceil", "np.ceil", (*ndarray.Array).Ceil),
		un("Floor", "np.floor", (*ndarray.Array).Floor),
		un("Round", "np.round", (*ndarray.Array).Round),
		un("Conj", "np.conj", (*ndarray.Array).Conj),
		un("Real", "np.real", (*ndarray.Array).Real),
		un("Imag", "np.imag", (*ndarray.Array).Imag),
		un("Angle", "np.angle", (*ndarray.Array).Angle),
		un("LogicalNot", "np.logical_not", (*ndarray.Array).LogicalNot),
		un("Copy", "lambda x: x.copy()", (*ndarray.Array).Copy),
		un("Flatten", "lambda x: x.flatten()", (*ndarray.Array).Flatten),
		un("Ravel", "lambda x: x.ravel()", (*ndarray.Array).Ravel),
		un("Power", "lambda x: np.power(x, 2.5)", func(a *ndarray.Array) *ndarray.Array { return a.Power(2.5) }),
		un("CumSumFlat", "np.cumsum", (*ndarray.Array).CumSumFlat),
		un("Nonzero", "np.flatnonzero", (*ndarray.Array).Nonzero),
		un("Repr", "repr", func(a *ndarray.Array) *ndarray.Array { sinkF = float64(len(a.Repr())); return a }),
		un("AsType", "lambda x: x.astype(np.float32)", func(a *ndarray.Array) *ndarray.Array { return a.AsType(ndarray.Float32) }),
		bin("Add", "np.add", (*ndarray.Array).Add),
		bin("Sub", "np.subtract", (*ndarray.Array).Sub),
		bin("Mul", "np.multiply", (*ndarray.Array).Mul),
		bin("Div", "np.divide", (*ndarray.Array).Div),
		bin("Maximum", "np.maximum", (*ndarray.Array).Maximum),
		bin("Minimum", "np.minimum", (*ndarray.Array).Minimum),
		bin("Equal", "np.equal", (*ndarray.Array).Equal),
		bin("NotEqual", "np.not_equal", (*ndarray.Array).NotEqual),
		bin("Less", "np.less", (*ndarray.Array).Less),
		bin("LessEqual", "np.less_equal", (*ndarray.Array).LessEqual),
		bin("Greater", "np.greater", (*ndarray.Array).Greater),
		bin("GreaterEqual", "np.greater_equal", (*ndarray.Array).GreaterEqual),
		bin("LogicalAnd", "np.logical_and", (*ndarray.Array).LogicalAnd),
		bin("LogicalOr", "np.logical_or", (*ndarray.Array).LogicalOr),
		bin("LogicalXor", "np.logical_xor", (*ndarray.Array).LogicalXor),
		bin("Inner", "np.inner", (*ndarray.Array).Inner),
		into("AddInto", "np.add", func(a, o, b *ndarray.Array) error { return a.AddInto(o, b) }),
		into("SubInto", "np.subtract", func(a, o, b *ndarray.Array) error { return a.SubInto(o, b) }),
		into("MulInto", "np.multiply", func(a, o, b *ndarray.Array) error { return a.MulInto(o, b) }),
		into("DivInto", "np.divide", func(a, o, b *ndarray.Array) error { return a.DivInto(o, b) }),
		scal("AddScalar", "+", (*ndarray.Array).AddScalar),
		scal("SubScalar", "-", (*ndarray.Array).SubScalar),
		scal("MulScalar", "*", (*ndarray.Array).MulScalar),
		scal("DivScalar", "/", (*ndarray.Array).DivScalar),
		{name: "Array.SqrtInto", fn: "Array.SqrtInto", ref: "numpy np.sqrt (out= preallocated)", args: []ds{dX},
			call: fmt.Sprintf("(lambda o: lambda x: np.sqrt(x, out=o))(np.empty(%d))", nvec),
			body: func(a []*ndarray.Array) func() {
				out := must(ndarray.New(nvec))
				return func() { chk(a[0].SqrtInto(out)) }
			}},
		{name: "Array.Clip", fn: "Array.Clip", ref: "numpy np.clip", args: []ds{dX}, call: "lambda x: np.clip(x, 1.0, 3.0)",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].Clip(1, 3)) } }},
		{name: "Array.Dot", fn: "Array.Dot", ref: "numpy np.dot, 128 x 128 (OpenBLAS, one thread)", args: []ds{dA, dB}, call: "np.dot",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].Dot(a[1])) } }},
		{name: "Array.MatMul", fn: "Array.MatMul", ref: "numpy np.matmul, 128 x 128 (OpenBLAS, one thread)", args: []ds{dA, dB}, call: "np.matmul",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].MatMul(a[1])) } }},
		{name: "Array.Outer", fn: "Array.Outer", ref: "numpy np.outer, 1000 x 1000", args: []ds{dX1, dY1}, call: "np.outer",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = a[0].Outer(a[1]) } }},
		{name: "Array.Transpose", fn: "Array.Transpose", ref: "numpy x.T (a view; Go may copy)", args: []ds{dA}, call: "lambda x: x.T",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = a[0].Transpose() } }},
		{name: "Array.Reshape", fn: "Array.Reshape", ref: "numpy x.reshape (a view)", args: []ds{dX}, call: "lambda x: x.reshape(100, 1000)",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].Reshape(100, 1000)) } }},
		{name: "Array.ExpandDims", fn: "Array.ExpandDims", ref: "numpy np.expand_dims", args: []ds{dX}, call: "lambda x: np.expand_dims(x, 0)",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].ExpandDims(0)) } }},
		{name: "Array.Squeeze", fn: "Array.Squeeze", ref: "numpy np.squeeze", args: []ds{dreshape(dX, 1, nvec)}, call: "np.squeeze",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].Squeeze()) } }},
		{name: "Array.Slice", fn: "Array.Slice", ref: "numpy x[0:n:2] (a view)", args: []ds{dX}, call: "lambda x: x[0:100000:2]",
			body: func(a []*ndarray.Array) func() {
				return func() { sinkA = must(a[0].Slice(ndarray.Rng(0, nvec, 2))) }
			}},
		{name: "Array.Take", fn: "Array.Take", ref: "numpy np.take, 1000 indices", args: []ds{dX},
			call: "(lambda idx: lambda x: np.take(x, idx))((np.arange(1000) * 7919) % 100000)",
			body: func(a []*ndarray.Array) func() {
				idx := make([]int, 1000)
				for i := range idx {
					idx[i] = (i * 7919) % nvec
				}
				return func() { sinkA = must(a[0].Take(idx...)) }
			}},
		{name: "Array.MaskSelect", fn: "Array.MaskSelect", ref: "numpy x[mask] (mask x > 2 built outside the call)", args: []ds{dX},
			call: "(lambda mk: lambda x: x[mk])(" + maskPy + ")",
			body: func(a []*ndarray.Array) func() {
				mask := must(a[0].Greater(ndarray.Scalar(2.0)))
				return func() { sinkA = must(a[0].MaskSelect(mask)) }
			}},
		{name: "Where", fn: "Where", ref: "numpy np.where (mask x > 2 built outside the call)", args: []ds{dX, dY},
			call: "(lambda mk: lambda x, y: np.where(mk, x, y))(" + maskPy + ")",
			body: func(a []*ndarray.Array) func() {
				mask := must(a[0].Greater(ndarray.Scalar(2.0)))
				return func() { sinkA = must(ndarray.Where(mask, a[0], a[1])) }
			}},
		{name: "Concatenate", fn: "Concatenate", ref: "numpy np.concatenate", args: []ds{dX, dY}, call: "lambda x, y: np.concatenate((x, y))",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(ndarray.Concatenate(a, 0)) } }},
		{name: "Stack", fn: "Stack", ref: "numpy np.stack", args: []ds{dX, dY}, call: "lambda x, y: np.stack((x, y))",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(ndarray.Stack(a, 0)) } }},
		{name: "HStack", fn: "HStack", ref: "numpy np.hstack", args: []ds{dX, dY}, call: "lambda x, y: np.hstack((x, y))",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(ndarray.HStack(a)) } }},
		{name: "VStack", fn: "VStack", ref: "numpy np.vstack", args: []ds{dX, dY}, call: "lambda x, y: np.vstack((x, y))",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(ndarray.VStack(a)) } }},
		// Reductions to a scalar.
		{name: "Array.Sum", fn: "Array.Sum", ref: "numpy np.sum", args: []ds{dX}, call: "np.sum",
			body: func(a []*ndarray.Array) func() { return func() { sinkF = a[0].Sum() } }},
		{name: "Array.SumAll", fn: "Array.SumAll", ref: "numpy np.sum", args: []ds{dX}, call: "np.sum",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = a[0].SumAll() } }},
		{name: "Array.Prod", fn: "Array.Prod", ref: "numpy np.prod", args: []ds{dP}, call: "np.prod",
			body: func(a []*ndarray.Array) func() { return func() { sinkF = a[0].Prod() } }},
		{name: "Array.ProdAll", fn: "Array.ProdAll", ref: "numpy np.prod", args: []ds{dP}, call: "np.prod",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = a[0].ProdAll() } }},
		{name: "Array.Mean", fn: "Array.Mean", ref: "numpy np.mean", args: []ds{dX}, call: "np.mean",
			body: func(a []*ndarray.Array) func() {
				return func() { v, err := a[0].Mean(); chk(err); sinkF = v }
			}},
		{name: "Array.Max", fn: "Array.Max", ref: "numpy np.max", args: []ds{dX}, call: "np.max",
			body: func(a []*ndarray.Array) func() {
				return func() { v, err := a[0].Max(); chk(err); sinkF = v }
			}},
		{name: "Array.Min", fn: "Array.Min", ref: "numpy np.min", args: []ds{dX}, call: "np.min",
			body: func(a []*ndarray.Array) func() {
				return func() { v, err := a[0].Min(); chk(err); sinkF = v }
			}},
		{name: "Array.ArgMax", fn: "Array.ArgMax", ref: "numpy np.argmax", args: []ds{dX}, call: "np.argmax",
			body: func(a []*ndarray.Array) func() {
				return func() { v, err := a[0].ArgMax(); chk(err); sinkI = v }
			}},
		{name: "Array.ArgMin", fn: "Array.ArgMin", ref: "numpy np.argmin", args: []ds{dX}, call: "np.argmin",
			body: func(a []*ndarray.Array) func() {
				return func() { v, err := a[0].ArgMin(); chk(err); sinkI = v }
			}},
		{name: "Array.All", fn: "Array.All", ref: "numpy np.all", args: []ds{dX}, call: "np.all",
			body: func(a []*ndarray.Array) func() { return func() { sinkB = a[0].All() } }},
		{name: "Array.Any", fn: "Array.Any", ref: "numpy np.any", args: []ds{dX}, call: "np.any",
			body: func(a []*ndarray.Array) func() { return func() { sinkB = a[0].Any() } }},
		{name: "Array.CumProdFlat", fn: "Array.CumProdFlat", ref: "numpy np.cumprod", args: []ds{dP}, call: "np.cumprod",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = a[0].CumProdFlat() } }},
	}
	rs = append(rs, axisRows("SumAxis", "np.sum", dXM, (*ndarray.Array).SumAxis)...)
	rs = append(rs, axisRows("ProdAxis", "np.prod", dPM, (*ndarray.Array).ProdAxis)...)
	rs = append(rs, axisRows("MeanAxis", "np.mean", dXM, (*ndarray.Array).MeanAxis)...)
	rs = append(rs, axisRows("MaxAxis", "np.max", dXM, (*ndarray.Array).MaxAxis)...)
	rs = append(rs, axisRows("MinAxis", "np.min", dXM, (*ndarray.Array).MinAxis)...)
	rs = append(rs, axisRows("ArgMaxAxis", "np.argmax", dXM, (*ndarray.Array).ArgMaxAxis)...)
	rs = append(rs, axisRows("ArgMinAxis", "np.argmin", dXM, (*ndarray.Array).ArgMinAxis)...)
	for ax := 0; ax < 2; ax++ {
		rs = append(rs,
			row{name: fmt.Sprintf("Array.CumSum/axis%d", ax), fn: "Array.CumSum", ref: fmt.Sprintf("numpy np.cumsum(axis=%d)", ax),
				args: []ds{dXM}, call: fmt.Sprintf("lambda x: np.cumsum(x, axis=%d)", ax),
				body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].CumSum(ax)) } }},
			row{name: fmt.Sprintf("Array.CumProd/axis%d", ax), fn: "Array.CumProd", ref: fmt.Sprintf("numpy np.cumprod(axis=%d)", ax),
				args: []ds{dPM}, call: fmt.Sprintf("lambda x: np.cumprod(x, axis=%d)", ax),
				body: func(a []*ndarray.Array) func() { return func() { sinkA = must(a[0].CumProd(ax)) } }})
	}
	return rs
}

// ctorRows are the constructors: nothing to read, the reference builds the same array.
func ctorRows() []row {
	f := func(name, py string, g func() *ndarray.Array, args ...ds) row {
		return row{name: name, fn: name, ref: "numpy " + py, args: args, call: "lambda *a: " + py,
			body: func(a []*ndarray.Array) func() { return func() { sinkA = g() } }}
	}
	const eye = 512
	return []row{
		f("Arange", "np.arange(0.0, 100000.0, 1.0)", func() *ndarray.Array { return must(ndarray.Arange(0, nvec, 1)) }),
		f("Eye", "np.eye(512)", func() *ndarray.Array { return must(ndarray.Eye(eye, eye, 0)) }),
		f("Identity", "np.identity(512)", func() *ndarray.Array { return must(ndarray.Identity(eye)) }),
		f("Full", "np.full(100000, 1.5)", func() *ndarray.Array { return must(ndarray.Full(1.5, nvec)) }),
		f("FullOf", "np.full(100000, 1.5, dtype=np.float32)", func() *ndarray.Array { return must(ndarray.FullOf(ndarray.Float32, 1.5, nvec)) }),
		f("Linspace", "np.linspace(0.0, 1.0, 100000)", func() *ndarray.Array { return must(ndarray.Linspace(0, 1, nvec)) }),
		f("New", "np.zeros(100000)", func() *ndarray.Array { return must(ndarray.New(nvec)) }),
		f("Zeros", "np.zeros(100000)", func() *ndarray.Array { return must(ndarray.Zeros(nvec)) }),
		f("Ones", "np.ones(100000)", func() *ndarray.Array { return must(ndarray.Ones(nvec)) }),
		f("OnesOf", "np.ones(100000, dtype=np.float32)", func() *ndarray.Array { return must(ndarray.OnesOf(ndarray.Float32, nvec)) }),
		f("ZerosOf", "np.zeros(100000, dtype=np.float32)", func() *ndarray.Array { return must(ndarray.ZerosOf(ndarray.Float32, nvec)) }),
		f("Scalar", "np.array(1.5)", func() *ndarray.Array { return ndarray.Scalar(1.5) }),
		{name: "FromData", fn: "FromData", ref: "numpy np.asarray (wraps, no copy)", args: []ds{dX}, call: "np.asarray",
			body: func(a []*ndarray.Array) func() {
				d, _ := ndarray.Data[float64](a[0])
				return func() { sinkA = must(ndarray.FromData(d, nvec)) }
			}},
		{name: "FromSlice", fn: "FromSlice", ref: "numpy np.asarray (wraps, no copy)", args: []ds{dX}, call: "np.asarray",
			body: func(a []*ndarray.Array) func() {
				d, _ := ndarray.Data[float64](a[0])
				return func() { sinkA = must(ndarray.FromSlice(d, nvec)) }
			}},
	}
}
