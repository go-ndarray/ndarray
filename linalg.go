package ndarray

import (
	"fmt"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// ErrLinalg is returned when operands have shapes incompatible with the
// requested linear-algebra operation (e.g. a dimension mismatch in matrix
// multiply).
var ErrLinalg = fmt.Errorf("ndarray: incompatible shapes for linear algebra")

// matmul2D multiplies an (m x k) by a (k x n), returning a fresh (m x n)
// array. Either operand may be any 2-D view: the GEMM packs it through its
// strides (matView), so nothing is materialised first.
func matmul2D(a, b *Array, m, k, n int) *Array {
	dst := a.ws.alloc(m*n, true) // MatMulStridedP accumulates into dst
	kernels.MatMulStridedP(dst, a.matView(), b.matView(), m, k, n)
	shape := []int{m, n}
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape), ws: a.ws}
}

// matView is a 2-D array as the GEMM's strided operand view. The packing
// reads it in place, so a transposed or sliced operand (Inner's bᵀ, a[:, ::2])
// is never materialised into a contiguous copy first.
func (a *Array) matView() kernels.Mat {
	return kernels.Mat{Data: a.data, Off: a.offset, RS: a.strides[0], CS: a.strides[1]}
}

// MatMul returns the matrix product of two 2-D arrays, matching numpy.matmul
// (the @ operator) for the 2-D case: a is (m x k), b is (k x n), the result is
// (m x n). Both operands must be 2-D and the inner dimensions must agree.
func (a *Array) MatMul(b *Array) (*Array, error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, fmt.Errorf("%w: MatMul needs two 2-D arrays, got %d-D and %d-D",
			ErrLinalg, len(a.shape), len(b.shape))
	}
	if a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("%w: MatMul %v x %v inner dims differ",
			ErrLinalg, a.shape, b.shape)
	}
	// (m, 0) @ (0, n) holds no data, yet its m*n result can overflow.
	if err := validateShape([]int{a.shape[0], b.shape[1]}); err != nil {
		return nil, err
	}
	if a.dtype != Float64 || b.dtype != Float64 {
		return matmulTyped(a, b), nil
	}
	return matmul2D(a, b, a.shape[0], a.shape[1], b.shape[1]), nil
}

// matmulTyped is MatMul for operands of other dtypes, in their common dtype.
//
// Float32 runs on the packed float32 GEMM where there is a float32 kernel
// (arm64, ppc64le, and amd64 with FMA): it accumulates in float32, as NumPy's BLAS
// sgemm does, at twice the SIMD lanes of float64. Elsewhere it runs on the
// float64 GEMM: the operands are widened (exact, O(n²)), multiplied and
// accumulated in float64, and each element is rounded once to float32, which
// is at least as accurate; on a POWER9 that was 9x faster at 256² than the
// triple loop below, which the other dtypes keep.
func matmulTyped(a, b *Array) *Array {
	dt := ResultType(a.dtype, b.dtype)
	switch dt {
	case Float64:
		return a.toFloat64().matmulFloat64(b.toFloat64())
	case Float32:
		if useGemm32 {
			return matmulFloat32(a, b)
		}
		return a.toFloat64().matmulFloat64(b.toFloat64()).AsType(Float32)
	}
	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	d := makeStore(dt, m*n)
	matmulStore(d, a.operandStore(dt, a.shape), b.operandStore(dt, b.shape), m, k, n)
	return a.result(d, []int{m, n})
}

// useGemm32 selects the float32 GEMM. It is a var so tests can force the
// widening path on hardware that has the kernel (never the reverse).
var useGemm32 = kernels.HasGemm32()

// matmulFloat32 multiplies two validated 2-D operands whose common dtype is
// Float32 on the float32 GEMM. A Float32 operand is read in place through its
// strides, like matView; another dtype is converted first.
func matmulFloat32(a, b *Array) *Array {
	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	dst := make([]float32, m*n) // MatMul32StridedP accumulates into dst
	kernels.MatMul32StridedP(dst, a.matView32(), b.matView32(), m, k, n)
	return a.result(dst, []int{m, n})
}

// matView32 is matView for an operand of the float32 GEMM.
func (a *Array) matView32() kernels.Mat32 {
	if a.dtype != Float32 {
		return kernels.Mat32{Data: a.operandStore(Float32, a.shape).([]float32), RS: a.shape[1], CS: 1}
	}
	return kernels.Mat32{Data: typed[float32](a), Off: a.offset, RS: a.strides[0], CS: a.strides[1]}
}

// matmulFloat64 is matmul2D on two validated 2-D Float64 arrays.
func (a *Array) matmulFloat64(b *Array) *Array {
	return matmul2D(a, b, a.shape[0], a.shape[1], b.shape[1])
}

// Dot returns the dot product following numpy.dot for 1-D and 2-D operands:
//
//   - 1-D · 1-D  -> a 0-d (scalar) array, the inner product.
//   - 2-D · 2-D  -> matrix multiply (m x k)·(k x n) = (m x n).
//   - 2-D · 1-D  -> matrix-vector (m x k)·(k,) = (m,).
//   - 1-D · 2-D  -> vector-matrix (k,)·(k x n) = (n,).
//
// Higher-rank operands are rejected (the general tensordot is a later phase).
func (a *Array) Dot(b *Array) (*Array, error) {
	na, nb := len(a.shape), len(b.shape)
	if (a.dtype != Float64 || b.dtype != Float64) && na >= 1 && na <= 2 && nb >= 1 && nb <= 2 {
		return a.dotTyped(b)
	}
	switch {
	case na == 1 && nb == 1:
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot vectors of length %d and %d",
				ErrLinalg, a.shape[0], b.shape[0])
		}
		s := kernels.Dot1DP(a.contiguousData(), b.contiguousData())
		return &Array{data: []float64{s}, shape: []int{}, strides: []int{}, ws: a.ws}, nil

	case na == 2 && nb == 2:
		return a.MatMul(b)

	case na == 2 && nb == 1:
		// (m x k) · (k,) = (m,): m independent contiguous row-dots — far cheaper
		// than a single-column GEMM through the packer.
		if a.shape[1] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot %v x %v inner dims differ",
				ErrLinalg, a.shape, b.shape)
		}
		m, k := a.shape[0], a.shape[1]
		dst := a.alloc(m, false)
		kernels.MatVecP(dst, a.contiguousData(), b.contiguousData(), m, k)
		return &Array{data: dst, shape: []int{m}, strides: []int{1}, ws: a.ws}, nil

	case na == 1 && nb == 2:
		// (k,) · (k x n) = (n,): cache-friendly row-streaming accumulation, no
		// transpose and no GEMM packing for a single result row.
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot %v x %v inner dims differ",
				ErrLinalg, a.shape, b.shape)
		}
		k, n := b.shape[0], b.shape[1]
		dst := a.alloc(n, true) // VecMatP accumulates into dst
		kernels.VecMatP(dst, a.contiguousData(), b.contiguousData(), k, n)
		return &Array{data: dst, shape: []int{n}, strides: []int{1}, ws: a.ws}, nil

	default:
		return nil, fmt.Errorf("%w: Dot supports 1-D and 2-D operands, got %d-D and %d-D",
			ErrLinalg, na, nb)
	}
}

// Inner returns the inner product over the last axes, matching numpy.inner for
// 1-D and 2-D operands: a sum-product over the last axis, which must have the
// same length in both operands.
//
//   - 1-D · 1-D -> scalar (same as Dot).
//   - 2-D · 2-D -> (m x p) where a is (m x k), b is (p x k): out[i,j] is the
//     dot of row i of a with row j of b.
func (a *Array) Inner(b *Array) (*Array, error) {
	na, nb := len(a.shape), len(b.shape)
	if na == 1 && nb == 1 {
		return a.Dot(b)
	}
	if na == 2 && nb == 2 {
		if a.shape[1] != b.shape[1] {
			return nil, fmt.Errorf("%w: Inner %v x %v last dims differ",
				ErrLinalg, a.shape, b.shape)
		}
		// inner(a, b) == a · bᵀ.
		return a.MatMul(b.Transpose())
	}
	return nil, fmt.Errorf("%w: Inner supports 1-D and 2-D operands, got %d-D and %d-D",
		ErrLinalg, na, nb)
}

// Outer returns the outer product of two arrays, matching numpy.outer: both
// operands are flattened to 1-D vectors u (length m) and v (length n), and the
// result is the (m x n) array out[i,j] = u[i]*v[j].
func (a *Array) Outer(b *Array) *Array {
	if a.dtype != Float64 || b.dtype != Float64 {
		u, _ := a.Ravel().Reshape(-1, 1) // a column and a row: these reshapes
		v, _ := b.Ravel().Reshape(1, -1) // and their product cannot fail
		r, _ := u.Mul(v)
		return r
	}
	u := a.contiguousData()
	v := b.contiguousData()
	m, n := len(u), len(v)
	dst := a.alloc(m*n, false)
	for i := 0; i < m; i++ {
		ui := u[i]
		row := dst[i*n : i*n+n]
		for j := 0; j < n; j++ {
			row[j] = ui * v[j]
		}
	}
	shape := []int{m, n}
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape), ws: a.ws}
}

// dotTyped is Dot for 1-D and 2-D operands of other dtypes: vectors are made
// matrices, multiplied, and the added axes dropped again.
func (a *Array) dotTyped(b *Array) (*Array, error) {
	x, y := a, b
	if len(a.shape) == 1 {
		x = a.view([]int{1, a.shape[0]}, []int{0, a.strides[0]}, a.offset)
	}
	if len(b.shape) == 1 {
		y = b.view([]int{b.shape[0], 1}, []int{b.strides[0], 0}, b.offset)
	}
	r, err := x.MatMul(y)
	if err != nil {
		return nil, fmt.Errorf("%w: Dot %v x %v inner dims differ", ErrLinalg, a.shape, b.shape)
	}
	var shape []int
	if len(a.shape) == 2 {
		shape = append(shape, a.shape[0])
	}
	if len(b.shape) == 2 {
		shape = append(shape, b.shape[1])
	}
	return r.Reshape(shape...)
}
