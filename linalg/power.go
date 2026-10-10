package linalg

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

// matmul multiplies two arrays of the same shape (..., M, M) matrix by
// matrix, in their common dtype (the ndarray package's MatMul: the packed
// SIMD GEMM for float64).
func matmul(x, y *ndarray.Array) *ndarray.Array {
	s := x.Shape()
	if len(s) == 2 {
		r, _ := x.MatMul(y)
		return r
	}
	m := s[len(s)-1]
	cnt := prod(s[:len(s)-2])
	xr, _ := x.Reshape(cnt, m, m)
	yr, _ := y.Reshape(cnt, m, m)
	parts := make([]*ndarray.Array, cnt)
	for i := range parts {
		xi, _ := xr.Slice(ndarray.A(i), ndarray.All(), ndarray.All())
		yi, _ := yr.Slice(ndarray.A(i), ndarray.All(), ndarray.All())
		parts[i], _ = xi.MatMul(yi)
	}
	st, _ := ndarray.Stack(parts, 0)
	r, _ := st.Reshape(s...)
	return r
}

// MatrixPower is numpy.linalg.matrix_power: each square matrix of a
// (..., M, M) raised to the integer power n, by repeated squaring in
// numpy's order. n = 0 gives identities of a's dtype; a negative n inverts
// first (so the result is floating point, and a singular matrix gives
// ErrSingular). Integer matrices stay integer for n >= 0.
func MatrixPower(a *ndarray.Array, n int) (*ndarray.Array, error) {
	o, err := square(a, "MatrixPower")
	if err != nil {
		return nil, err
	}
	if n == 0 {
		m := o.m
		id := make([]float64, o.count()*m*m)
		for b := 0; b < o.count(); b++ {
			for i := 0; i < m; i++ {
				id[b*m*m+i*m+i] = 1
			}
		}
		r, _ := ndarray.FromSlice(id, a.Shape()...)
		return r.AsType(a.DType()), nil
	}
	if n < 0 {
		if a, err = Inv(a); err != nil {
			return nil, err
		}
		n = -n
	}
	if o.count() == 0 {
		return a.Copy(), nil
	}
	switch n {
	case 1:
		return a.Copy(), nil
	case 2:
		return matmul(a, a), nil
	case 3:
		return matmul(matmul(a, a), a), nil
	}
	var z, result *ndarray.Array
	for n > 0 {
		if z == nil {
			z = a
		} else {
			z = matmul(z, z)
		}
		bit := n % 2
		n /= 2
		if bit == 1 {
			if result == nil {
				result = z
			} else {
				result = matmul(result, z)
			}
		}
	}
	return result, nil
}

// MultiDot is numpy.linalg.multi_dot: the product of two or more arrays,
// evaluated in the order of fewest scalar multiplications (the classic
// matrix-chain dynamic programme, as numpy does). The first and last
// arrays may be vectors — a row and a column vector — the others must be
// 2-D. Two vectors give a 0-d result, one gives a vector.
func MultiDot(arrays ...*ndarray.Array) (*ndarray.Array, error) {
	n := len(arrays)
	if n < 2 {
		return nil, fmt.Errorf("%w: MultiDot needs at least two arrays, got %d", ErrArgs, n)
	}
	if n == 2 {
		r, err := arrays[0].Dot(arrays[1])
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrShape, err)
		}
		return r, nil
	}
	arr := append([]*ndarray.Array(nil), arrays...)
	firstVec, lastVec := arr[0].Ndim() == 1, arr[n-1].Ndim() == 1
	if firstVec {
		arr[0], _ = arr[0].Reshape(1, -1)
	}
	if lastVec {
		arr[n-1], _ = arr[n-1].Reshape(-1, 1)
	}
	for i, a := range arr {
		if a.Ndim() != 2 {
			return nil, fmt.Errorf("%w: MultiDot array %d is %d-D, not 2-D", ErrShape, i, a.Ndim())
		}
	}
	for i := 0; i+1 < n; i++ {
		if arr[i].Shape()[1] != arr[i+1].Shape()[0] {
			return nil, fmt.Errorf("%w: MultiDot arrays %d and %d: %v times %v", ErrShape, i, i+1, arr[i].Shape(), arr[i+1].Shape())
		}
	}
	var res *ndarray.Array
	if n == 3 {
		a0, a1b0 := arr[0].Shape()[0], arr[0].Shape()[1]
		b1c0, c1 := arr[2].Shape()[0], arr[2].Shape()[1]
		if a0*b1c0*(a1b0+c1) < a1b0*c1*(a0+b1c0) {
			res = dot(dot(arr[0], arr[1]), arr[2])
		} else {
			res = dot(arr[0], dot(arr[1], arr[2]))
		}
	} else {
		order := chainOrder(arr)
		res = chain(arr, order, 0, n-1)
	}
	switch {
	case firstVec && lastVec:
		r, _ := res.Reshape()
		return r, nil
	case firstVec || lastVec:
		return res.Ravel(), nil
	}
	return res, nil
}

func dot(a, b *ndarray.Array) *ndarray.Array {
	r, _ := a.MatMul(b)
	return r
}

// chainOrder is numpy's _multi_dot_matrix_chain_order: s[i][j] is where the
// product of arrays i..j is best split.
func chainOrder(arr []*ndarray.Array) [][]int {
	n := len(arr)
	p := make([]float64, n+1)
	for i, a := range arr {
		p[i] = float64(a.Shape()[0])
	}
	p[n] = float64(arr[n-1].Shape()[1])
	m := make([][]float64, n)
	s := make([][]int, n)
	for i := range m {
		m[i] = make([]float64, n)
		s[i] = make([]int, n)
	}
	for l := 1; l < n; l++ {
		for i := 0; i < n-l; i++ {
			j := i + l
			m[i][j] = -1
			for k := i; k < j; k++ {
				q := m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
				if m[i][j] < 0 || q < m[i][j] {
					m[i][j], s[i][j] = q, k
				}
			}
		}
	}
	return s
}

func chain(arr []*ndarray.Array, s [][]int, i, j int) *ndarray.Array {
	if i == j {
		return arr[i]
	}
	return dot(chain(arr, s, i, s[i][j]), chain(arr, s, s[i][j]+1, j))
}
