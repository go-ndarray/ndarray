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
	src := a.contiguousData()
	dst := make([]float64, len(src))
	kernels.SqrtP(dst, src)
	cp := append([]int(nil), a.shape...)
	return &Array{data: dst, shape: cp, strides: rowMajorStrides(cp)}
}

// SqrtInto writes the elementwise square root of a into the caller-provided
// contiguous out array (no allocation), the analogue of np.sqrt(a, out=out) and
// the parity path for small arrays where Go's per-op make would otherwise cost
// more than NumPy's cached temp buffer. out must be contiguous and the same
// shape as a; it may alias a (each index is read before it is written). It
// routes through the same SIMD sqrt kernel as Sqrt (packed SQRTPD on amd64, the
// FSQRTD-intrinsic scalar loop on arm64/others), bit-identical to math.Sqrt.
func (a *Array) SqrtInto(out *Array) error {
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
func (a *Array) Exp() *Array { return a.Map(math.Exp) }

// Log returns the elementwise natural logarithm.
func (a *Array) Log() *Array { return a.Map(math.Log) }

// Log2 returns the elementwise base-2 logarithm, within 2 ULP of numpy's.
func (a *Array) Log2() *Array { return a.Map(log2) }

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

// Log10 returns the elementwise base-10 logarithm.
func (a *Array) Log10() *Array { return a.Map(math.Log10) }

// Sin returns the elementwise sine (radians).
func (a *Array) Sin() *Array { return a.Map(math.Sin) }

// Cos returns the elementwise cosine (radians).
func (a *Array) Cos() *Array { return a.Map(math.Cos) }

// Tan returns the elementwise tangent (radians).
func (a *Array) Tan() *Array { return a.Map(math.Tan) }

// Floor returns the elementwise floor (greatest integer <= x).
func (a *Array) Floor() *Array { return a.Map(math.Floor) }

// Ceil returns the elementwise ceiling (least integer >= x).
func (a *Array) Ceil() *Array { return a.Map(math.Ceil) }

// Round returns the elementwise round-half-away-from-zero, matching math.Round.
func (a *Array) Round() *Array { return a.Map(math.Round) }

// Square returns the elementwise square x*x.
func (a *Array) Square() *Array { return a.Map(func(x float64) float64 { return x * x }) }

// Power returns the elementwise a**p (every element raised to p).
func (a *Array) Power(p float64) *Array {
	return a.Map(func(x float64) float64 { return math.Pow(x, p) })
}

// Comparison ufuncs. Each compares two arrays elementwise with NumPy
// broadcasting and returns a new array whose elements are 1.0 where the
// relation holds and 0.0 otherwise. (A dedicated bool dtype is a later phase;
// until then masks are float 0/1, which compose directly with the arithmetic
// ufuncs — e.g. mask.Mul(x) zeroes the unselected elements.)

// Equal returns the elementwise a == b mask.
func (a *Array) Equal(b *Array) (*Array, error) { return a.binOp(b, kernels.Equal) }

// NotEqual returns the elementwise a != b mask.
func (a *Array) NotEqual(b *Array) (*Array, error) { return a.binOp(b, kernels.NotEqual) }

// Greater returns the elementwise a > b mask.
func (a *Array) Greater(b *Array) (*Array, error) { return a.binOp(b, kernels.Greater) }

// GreaterEqual returns the elementwise a >= b mask.
func (a *Array) GreaterEqual(b *Array) (*Array, error) {
	return a.binOp(b, kernels.GreaterEqual)
}

// Less returns the elementwise a < b mask.
func (a *Array) Less(b *Array) (*Array, error) { return a.binOp(b, kernels.Less) }

// LessEqual returns the elementwise a <= b mask.
func (a *Array) LessEqual(b *Array) (*Array, error) { return a.binOp(b, kernels.LessEqual) }

// Maximum returns the elementwise pairwise maximum of a and b (broadcasting).
func (a *Array) Maximum(b *Array) (*Array, error) { return a.binOp(b, kernels.Maximum) }

// Minimum returns the elementwise pairwise minimum of a and b (broadcasting).
func (a *Array) Minimum(b *Array) (*Array, error) { return a.binOp(b, kernels.Minimum) }
