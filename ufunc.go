package ndarray

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// Elementwise math ufuncs. Each applies a math function to every element and
// returns a new contiguous array of the same shape, routed through the Map
// seam so a future SIMD kernel can replace the scalar inner loop. Domain errors
// (e.g. Sqrt or Log of a negative) follow Go's math package and yield NaN,
// matching NumPy's behaviour (NumPy additionally warns; this package does not).

// Sqrt returns the elementwise non-negative square root. It is routed through
// the dedicated SIMD sqrt kernel (packed SQRTPD on amd64; the intrinsic FSQRTD
// scalar loop on arm64/others) rather than the generic Map, because passing
// math.Sqrt as a func(float64)float64 blocks both the compiler's FSQRTD
// intrinsic and the packed kernel. The result is bit-identical to a scalar
// math.Sqrt loop (sqrt(-x)=NaN, sqrt(+Inf)=+Inf, matching NumPy).
func (a *Array) Sqrt() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uSqrt, (*Array).Sqrt)
	}
	src := a.contiguousData()
	dst := a.alloc(len(src), false)
	kernels.SqrtP(dst, src)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp), ws: a.ws}
}

// SqrtInto writes the elementwise square root of a into the caller-provided
// contiguous out array (no allocation), the analogue of np.sqrt(a, out=out) and
// the parity path for small arrays where Go's per-op make would otherwise cost
// more than NumPy's cached temp buffer. out must be contiguous and the same
// shape as a; it may alias a (each index is read before it is written). It
// routes through the same SIMD sqrt kernel as Sqrt (packed SQRTPD on amd64, the
// FSQRTD-intrinsic scalar loop on arm64/others), bit-identical to math.Sqrt.
func (a *Array) SqrtInto(out *Array) error {
	if a.dtype != Float64 || out.dtype != Float64 {
		return fmt.Errorf("%w: SqrtInto needs float64 arrays; use Sqrt", ErrDType)
	}
	if !out.isContiguous() {
		return fmt.Errorf("%w: out must be contiguous", ErrBroadcast)
	}
	if !sameShape(out.shape, a.shape) {
		return fmt.Errorf("%w: out shape %v != a shape %v",
			ErrBroadcast, out.shape, a.shape)
	}
	kernels.SqrtP(out.data, a.contiguousData())
	return nil
}

// Exp returns the elementwise base-e exponential.
//
// It runs a table-driven port of Arm's optimized-routines exp (0.51 ULP worst
// case, measured), split across cores: several times faster than math.Exp per
// element, and finite up to ln(MaxFloat64) on amd64, where math.Exp returns
// +Inf from x ~ 709.436 (golang/go#81995; Go 1.26.8 and 1.27.1).
func (a *Array) Exp() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uExp, (*Array).Exp)
	}
	src := a.contiguousData()
	dst := a.alloc(len(src), false)
	kernels.ExpP(dst, src)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp), ws: a.ws}
}

// Log returns the elementwise natural logarithm.
//
// It runs a table-driven port of Arm's optimized-routines log (0.51 ULP worst
// case, measured), split across cores: about twice as fast as math.Log, and
// correct for subnormal inputs on amd64, where math.Log is not (log(5e-324)
// gives -709.09 instead of -744.44; golang/go#56600).
func (a *Array) Log() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uLog, (*Array).Log)
	}
	return a.unary(kernels.LogP)
}

// Log2 returns the elementwise base-2 logarithm, within 2 ULP of numpy's.
func (a *Array) Log2() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uLog2, (*Array).Log2)
	}
	return a.Map(log2)
}

// log2 is math.Log2 without its cancellation near 1. math.Log2 computes
// log(frac)/ln2 + exp with frac in [0.5, 1), so for x just above 1 it adds
// about -1 and 1 and keeps few correct digits (log2(1+1e-10) had a relative
// error of 3e-7). Centring frac on 1, in [sqrt(1/2), sqrt(2)), makes exp 0
// there, and the result is as accurate as math.Log. Powers of two stay exact,
// and 0, negatives, ±Inf and NaN map as math.Log2 maps them.
func log2(x float64) float64 {
	frac, exp := math.Frexp(x)
	if frac < math.Sqrt2/2 {
		frac *= 2
		exp--
	}
	return math.Log(frac)*(1/math.Ln2) + float64(exp)
}

// Log10 returns the elementwise base-10 logarithm, log(x)*(1/ln10) as
// math.Log10 computes it, on the same log as Log (so subnormals are right).
func (a *Array) Log10() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uLog10, (*Array).Log10)
	}
	return a.unary(kernels.Log10P)
}

// unary applies a contiguous-slice kernel elementwise into a new array.
func (a *Array) unary(kernel func(dst, src []float64)) *Array {
	src := a.contiguousData()
	dst := a.alloc(len(src), false)
	kernel(dst, src)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp), ws: a.ws}
}

// Sin returns the elementwise sine (radians).
func (a *Array) Sin() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uSin, (*Array).Sin)
	}
	return a.Map(math.Sin)
}

// Cos returns the elementwise cosine (radians).
func (a *Array) Cos() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uCos, (*Array).Cos)
	}
	return a.Map(math.Cos)
}

// Tan returns the elementwise tangent (radians).
func (a *Array) Tan() *Array {
	if a.dtype != Float64 {
		return a.mathUnary(uTan, (*Array).Tan)
	}
	return a.Map(math.Tan)
}

// Floor returns the elementwise floor (greatest integer <= x).
func (a *Array) Floor() *Array {
	if a.dtype != Float64 {
		return a.exactUnary(uFloor, (*Array).Floor)
	}
	return a.Map(math.Floor)
}

// Ceil returns the elementwise ceiling (least integer >= x).
func (a *Array) Ceil() *Array {
	if a.dtype != Float64 {
		return a.exactUnary(uCeil, (*Array).Ceil)
	}
	return a.Map(math.Ceil)
}

// Round returns the elementwise round-half-away-from-zero, matching math.Round.
func (a *Array) Round() *Array {
	if a.dtype != Float64 {
		return a.exactUnary(uRound, (*Array).Round)
	}
	return a.Map(math.Round)
}

// Square returns the elementwise square x*x.
// Integers wrap and keep their dtype; a Bool array gives Int8, as in NumPy.
func (a *Array) Square() *Array {
	if a.dtype != Float64 {
		return a.intUnary(uSquare, (*Array).Square)
	}
	return a.Map(func(x float64) float64 { return x * x })
}

// Power returns the elementwise a**p (every element raised to p).
//
// p is a weak float: Float32 and complex arrays keep their dtype, integer
// and Bool arrays give Float64.
func (a *Array) Power(p float64) *Array {
	if a.dtype != Float64 {
		return a.power(p)
	}
	return a.Map(func(x float64) float64 { return math.Pow(x, p) })
}

// Comparison ufuncs. Each promotes both arrays to their common dtype,
// compares them elementwise with NumPy broadcasting, and returns a Bool array.
// Complex numbers order lexicographically, as in NumPy. A mask can be used
// in arithmetic directly (true is 1): mask.Mul(x) zeroes the unselected
// elements.

// Equal returns the elementwise a == b mask.
func (a *Array) Equal(b *Array) (*Array, error) { return a.compare(b, opEq) }

// NotEqual returns the elementwise a != b mask.
func (a *Array) NotEqual(b *Array) (*Array, error) { return a.compare(b, opNe) }

// Greater returns the elementwise a > b mask.
func (a *Array) Greater(b *Array) (*Array, error) { return a.compare(b, opGt) }

// GreaterEqual returns the elementwise a >= b mask.
func (a *Array) GreaterEqual(b *Array) (*Array, error) { return a.compare(b, opGe) }

// Less returns the elementwise a < b mask.
func (a *Array) Less(b *Array) (*Array, error) { return a.compare(b, opLt) }

// LessEqual returns the elementwise a <= b mask.
func (a *Array) LessEqual(b *Array) (*Array, error) { return a.compare(b, opLe) }

// Maximum returns the elementwise pairwise maximum of a and b (broadcasting).
func (a *Array) Maximum(b *Array) (*Array, error) { return a.arith(b, opMax, kernels.Maximum) }

// Minimum returns the elementwise pairwise minimum of a and b (broadcasting).
func (a *Array) Minimum(b *Array) (*Array, error) { return a.arith(b, opMin, kernels.Minimum) }
