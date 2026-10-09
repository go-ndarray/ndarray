package linalg

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// Ord is a norm order, numpy's ord argument. The zero value is numpy's
// None: the 2-norm of vectors and the Frobenius norm of matrices. Make a
// numeric order with P (P(math.Inf(1)) is numpy's inf); Fro and Nuc are
// numpy's 'fro' and 'nuc'.
type Ord struct {
	kind uint8
	p    float64
}

const (
	ordDefault uint8 = iota
	ordNum
	ordFro
	ordNuc
)

// The named matrix norms.
var (
	Fro = Ord{kind: ordFro} // the Frobenius norm
	Nuc = Ord{kind: ordNuc} // the nuclear norm, the sum of the singular values
)

// P is the numeric norm order p: for vectors (Σ|x|^p)^(1/p), with 0 the
// count of nonzeros and ±Inf the largest and smallest |x|; for matrices
// ±1, ±2 and ±Inf only.
func P(p float64) Ord { return Ord{kind: ordNum, p: p} }

func (o Ord) String() string {
	switch o.kind {
	case ordFro:
		return "fro"
	case ordNuc:
		return "nuc"
	case ordNum:
		return fmt.Sprint(o.p)
	}
	return "None"
}

// Norm is numpy.linalg.norm(x, ord, axis, keepdims).
//
// With axes nil: the default ord is the 2-norm of all of x flattened,
// whatever its rank; another ord takes x as a vector (1-D) or a matrix
// (2-D). With one axis, vector norms are taken along it; with two, matrix
// norms over them (the first indexes rows). keepdims keeps the reduced axes
// with length one. The result is float64, or float32 for float32 and
// complex64 inputs.
//
// The sums are accumulated in double precision, and the 2-norms are
// rescaled when the sum of squares would overflow or underflow, so a
// float32 norm, or a norm of huge or tiny values, can be more accurate than
// numpy's (which overflows to Inf where this returns the norm).
func Norm(x *ndarray.Array, ord Ord, axes []int, keepdims bool) (*ndarray.Array, error) {
	if x.DType().IsComplex() {
		return normT[complex128](x, ord, axes, keepdims)
	}
	return normT[float64](x, ord, axes, keepdims)
}

func normT[T scalar](x *ndarray.Array, ord Ord, axes []int, keepdims bool) (*ndarray.Array, error) {
	shape := x.Shape()
	nd := len(shape)
	_, single := precision(x)
	data := load[T](x)
	if axes == nil {
		if ord.kind == ordDefault || (ord.kind == ordFro && nd == 2) || (ord.kind == ordNum && ord.p == 2 && nd == 1) {
			var os []int
			if keepdims {
				os = ones(nd)
			}
			return wrap([]float64{nrm2(len(data), data, 1)}, os, single), nil
		}
		axes = make([]int, nd)
		for i := range axes {
			axes[i] = i
		}
	}
	ax := make([]int, len(axes))
	for i, a := range axes {
		if a < -nd || a >= nd {
			return nil, fmt.Errorf("%w: axis %d out of range for %d-D array", ErrArgs, a, nd)
		}
		ax[i] = (a + nd) % nd
	}
	var vals []float64
	var err error
	switch len(ax) {
	case 1:
		moved, rest := moveToEnd(data, shape, ax)
		l := shape[ax[0]]
		vals = make([]float64, prod(rest))
		for i := range vals {
			if vals[i], err = vecNorm(moved[i*l:(i+1)*l], ord); err != nil {
				return nil, err
			}
		}
	case 2:
		if ax[0] == ax[1] {
			return nil, fmt.Errorf("%w: duplicate axes %v", ErrArgs, axes)
		}
		moved, rest := moveToEnd(data, shape, ax)
		r, c := shape[ax[0]], shape[ax[1]]
		vals = make([]float64, prod(rest))
		if len(vals) == 0 {
			// numpy still rejects the order, and an empty reduction axis.
			_, err = matNorm(r, c, make([]T, r*c), ord)
		}
		for i := range vals {
			if vals[i], err = matNorm(r, c, moved[i*r*c:(i+1)*r*c], ord); err != nil {
				break
			}
		}
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%w: Norm over %d axes", ErrArgs, len(ax))
	}
	out := make([]int, 0, nd)
	for i, s := range shape {
		if reduced(ax, i) {
			if keepdims {
				out = append(out, 1)
			}
			continue
		}
		out = append(out, s)
	}
	return wrap(vals, out, single), nil
}

func ones(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func reduced(ax []int, i int) bool {
	for _, a := range ax {
		if a == i {
			return true
		}
	}
	return false
}

// moveToEnd returns the elements of the row-major array (data, shape)
// reordered so the axes ax come last, in their order, and the shape of the
// other axes.
func moveToEnd[T scalar](data []T, shape, ax []int) ([]T, []int) {
	var order, rest []int
	for i := range shape {
		if !reduced(ax, i) {
			order = append(order, i)
			rest = append(rest, shape[i])
		}
	}
	order = append(order, ax...)
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	out := make([]T, len(data))
	idx := make([]int, len(order))
	for lin := range out {
		pos := 0
		for k, a := range order {
			pos += idx[k] * strides[a]
		}
		out[lin] = data[pos]
		for k := len(order) - 1; k >= 0; k-- {
			idx[k]++
			if idx[k] < shape[order[k]] {
				break
			}
			idx[k] = 0
		}
	}
	return out, rest
}

// vecNorm is the vector norm of x.
func vecNorm[T scalar](x []T, ord Ord) (float64, error) {
	if ord.kind == ordDefault || (ord.kind == ordNum && ord.p == 2) {
		return nrm2(len(x), x, 1), nil
	}
	if ord.kind != ordNum {
		return 0, fmt.Errorf("%w: invalid norm order '%v' for vectors", ErrArgs, ord)
	}
	p := ord.p
	switch {
	case math.IsInf(p, 1):
		m := 0.0
		for _, v := range x {
			m = math.Max(m, abs(v))
		}
		return m, nil
	case math.IsInf(p, -1):
		if len(x) == 0 {
			return 0, fmt.Errorf("%w: the -Inf norm of an empty vector", ErrShape)
		}
		m := math.Inf(1)
		for _, v := range x {
			m = math.Min(m, abs(v))
		}
		return m, nil
	case p == 0:
		c := 0.0
		for _, v := range x {
			if v != 0 {
				c++
			}
		}
		return c, nil
	case p == 1:
		s := 0.0
		for _, v := range x {
			s += abs(v)
		}
		return s, nil
	}
	s := 0.0
	for _, v := range x {
		s += math.Pow(abs(v), p)
	}
	return math.Pow(s, 1/p), nil
}

// matNorm is the matrix norm of the r×c row-major x.
func matNorm[T scalar](r, c int, x []T, ord Ord) (float64, error) {
	switch ord.kind {
	case ordDefault, ordFro:
		return nrm2(len(x), x, 1), nil
	case ordNuc:
		s, err := svals(r, c, x)
		total := 0.0
		for _, v := range s {
			total += v
		}
		return total, err
	}
	switch p := ord.p; p {
	case 2, -2:
		s, err := svals(r, c, x)
		if err != nil || p == 2 {
			return first(s), err
		}
		if len(s) == 0 {
			return 0, fmt.Errorf("%w: the -2 norm of an empty matrix", ErrShape)
		}
		return s[len(s)-1], nil
	case 1, -1:
		if p == -1 && c == 0 {
			return 0, fmt.Errorf("%w: the -1 norm of a matrix without columns", ErrShape)
		}
		sums := make([]float64, c)
		for i := range x {
			sums[i%c] += abs(x[i])
		}
		return extreme(sums, p > 0), nil
	case math.Inf(1), math.Inf(-1):
		if p < 0 && r == 0 {
			return 0, fmt.Errorf("%w: the -Inf norm of a matrix without rows", ErrShape)
		}
		sums := make([]float64, r)
		for i := range x {
			sums[i/c] += abs(x[i])
		}
		return extreme(sums, p > 0), nil
	}
	return 0, fmt.Errorf("%w: invalid norm order '%v' for matrices", ErrArgs, ord)
}

// svals is the singular values of the r×c matrix x (left intact).
func svals[T scalar](r, c int, x []T) ([]float64, error) {
	s, _, _, ok := svdOne(r, c, append([]T(nil), x...), false, false)
	if !ok {
		return nil, ErrNoConvergence
	}
	return s, nil
}

func first(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	return s[0]
}

// extreme is the largest (or smallest) of s, 0 for an empty s.
func extreme(s []float64, largest bool) float64 {
	if len(s) == 0 {
		return 0
	}
	m := s[0]
	for _, v := range s[1:] {
		if largest {
			m = math.Max(m, v)
		} else {
			m = math.Min(m, v)
		}
	}
	return m
}
