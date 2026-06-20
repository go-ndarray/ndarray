package ndarray

import (
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

// Exp returns the elementwise base-e exponential.
func (a *Array) Exp() *Array { return a.Map(math.Exp) }

// Log returns the elementwise natural logarithm.
func (a *Array) Log() *Array { return a.Map(math.Log) }

// Log2 returns the elementwise base-2 logarithm.
func (a *Array) Log2() *Array { return a.Map(math.Log2) }

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
