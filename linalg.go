package ndarray

import (
	"fmt"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// ErrLinalg is returned when operands have shapes incompatible with the
// requested linear-algebra operation (e.g. a dimension mismatch in matrix
// multiply).
var ErrLinalg = fmt.Errorf("ndarray: incompatible shapes for linear algebra")

// matmul2D multiplies a contiguous (m x k) by a contiguous (k x n), returning a
// fresh (m x n) array. The operands are materialised so strided/transposed
// views work transparently.
func matmul2D(a, b *Array, m, k, n int) *Array {
	dst := make([]float64, m*n)
	kernels.MatMulP(dst, a.contiguousData(), b.contiguousData(), m, k, n)
	shape := []int{m, n}
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape)}
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
	return matmul2D(a, b, a.shape[0], a.shape[1], b.shape[1]), nil
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
	switch {
	case na == 1 && nb == 1:
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot vectors of length %d and %d",
				ErrLinalg, a.shape[0], b.shape[0])
		}
		s := kernels.Dot1D(a.materialize(), b.materialize())
		return &Array{data: []float64{s}, shape: []int{}, strides: []int{}}, nil

	case na == 2 && nb == 2:
		return a.MatMul(b)

	case na == 2 && nb == 1:
		// (m x k) · (k,) = (m,): treat the vector as (k x 1), then drop the axis.
		if a.shape[1] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot %v x %v inner dims differ",
				ErrLinalg, a.shape, b.shape)
		}
		r := matmul2D(a, b, a.shape[0], a.shape[1], 1)
		return r.Reshape(a.shape[0])

	case na == 1 && nb == 2:
		// (k,) · (k x n) = (n,): treat the vector as (1 x k), then drop the axis.
		if a.shape[0] != b.shape[0] {
			return nil, fmt.Errorf("%w: Dot %v x %v inner dims differ",
				ErrLinalg, a.shape, b.shape)
		}
		r := matmul2D(a, b, 1, a.shape[0], b.shape[1])
		return r.Reshape(b.shape[1])

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
	u := a.materialize()
	v := b.materialize()
	m, n := len(u), len(v)
	dst := make([]float64, m*n)
	for i := 0; i < m; i++ {
		ui := u[i]
		row := dst[i*n : i*n+n]
		for j := 0; j < n; j++ {
			row[j] = ui * v[j]
		}
	}
	shape := []int{m, n}
	return &Array{data: dst, shape: shape, strides: rowMajorStrides(shape)}
}
