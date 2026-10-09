package linalg

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/go-ndarray/ndarray"
)

// getrf factors the m×n row-major matrix a (leading dimension lda) in place
// as P·L·U with partial pivoting (LAPACK getrf): L unit lower triangular
// below the diagonal, U upper triangular on and above it. piv[j] is the row
// interchanged with row j at step j. It returns the index of the first
// exactly zero pivot, or -1: like LAPACK, the factorization is still
// completed then.
//
// The recursion (LAPACK getrf2) splits the columns in halves, so all but a
// vanishing fraction of the work is the GEMM update of the trailing block.
// Row interchanges swap whole rows at once, which in row-major storage is a
// contiguous copy and is equivalent to LAPACK's deferred laswp calls.
func getrf[T scalar](m, n int, a []T, lda int, piv []int) int {
	k := min(m, n)
	info := luRec(m, n, a, lda, piv, 0, k)
	if n > m {
		trsmLLU(m, n-m, a, lda, a[m:], lda)
	}
	return info
}

// luRec factors columns c0..c0+w-1, rows c0..m-1, of the m×ncol matrix a.
func luRec[T scalar](m, ncol int, a []T, lda int, piv []int, c0, w int) int {
	if w <= trsmLeaf {
		return luLeaf(m, ncol, a, lda, piv, c0, w)
	}
	w1 := w / 2
	info := luRec(m, ncol, a, lda, piv, c0, w1)
	c1 := c0 + w1
	trsmLLU(w1, w-w1, a[c0*lda+c0:], lda, a[c0*lda+c1:], lda)
	gemm(m-c1, w-w1, w1, -1, rm(a, c1*lda+c0, lda), rm(a, c0*lda+c1, lda), 1, a[c1*lda+c1:], lda)
	if i2 := luRec(m, ncol, a, lda, piv, c1, w-w1); info < 0 {
		info = i2
	}
	return info
}

// luLeaf is the unblocked right-looking factorization of a narrow panel.
func luLeaf[T scalar](m, ncol int, a []T, lda int, piv []int, c0, w int) int {
	info := -1
	end := c0 + w
	for j := c0; j < end; j++ {
		p, best := j, -1.0
		for i := j; i < m; i++ {
			if v := abs1(a[i*lda+j]); v > best {
				p, best = i, v
			}
		}
		piv[j] = p
		if a[p*lda+j] == 0 {
			if info < 0 {
				info = j
			}
		} else {
			if p != j {
				rj, rp := a[j*lda:j*lda+ncol], a[p*lda:p*lda+ncol]
				for c := range rj {
					rj[c], rp[c] = rp[c], rj[c]
				}
			}
			d := a[j*lda+j]
			if abs(d) >= safmin {
				r := 1 / d
				for i := j + 1; i < m; i++ {
					a[i*lda+j] *= r
				}
			} else {
				for i := j + 1; i < m; i++ {
					a[i*lda+j] /= d
				}
			}
		}
		rowj := a[j*lda+j+1 : j*lda+end]
		for i := j + 1; i < m; i++ {
			l := a[i*lda+j]
			if l == 0 {
				continue
			}
			ri := a[i*lda+j+1 : i*lda+end]
			for c, u := range rowj {
				ri[c] -= l * u
			}
		}
	}
	return info
}

// getrs solves A·X = B with the LU factors of the n×n A, B n×nrhs in place.
func getrs[T scalar](n, nrhs int, lu []T, piv []int, b []T) {
	for i := 0; i < n; i++ {
		if p := piv[i]; p != i {
			ri, rp := b[i*nrhs:i*nrhs+nrhs], b[p*nrhs:p*nrhs+nrhs]
			for c := range ri {
				ri[c], rp[c] = rp[c], ri[c]
			}
		}
	}
	trsmLLU(n, nrhs, lu, n, b, nrhs)
	trsmLUN(n, nrhs, lu, n, b, nrhs)
}

// Solve is numpy.linalg.solve: the solution X of A·X = B for square A.
//
// a is (..., M, M). b is either a vector (M,), solved against every matrix
// of a, giving (..., M); or a stack (..., M, K), whose leading dimensions
// broadcast against a's, giving (..., M, K). As in numpy 2, a b of two or
// more dimensions is always a stack of matrices. An exactly singular matrix
// gives ErrSingular.
func Solve(a, b *ndarray.Array) (*ndarray.Array, error) {
	oa, err := square(a, "Solve")
	if err != nil {
		return nil, err
	}
	bs := b.Shape()
	var bb []int
	k := 1
	vec := len(bs) == 1
	switch {
	case vec:
		if bs[0] != oa.n {
			return nil, fmt.Errorf("%w: Solve %v and %v: b has %d rows, a %d", ErrShape, a.Shape(), bs, bs[0], oa.n)
		}
	case len(bs) < 2:
		return nil, fmt.Errorf("%w: Solve needs b of at least one dimension", ErrShape)
	default:
		if bs[len(bs)-2] != oa.n {
			return nil, fmt.Errorf("%w: Solve %v and %v: b has %d rows, a %d", ErrShape, a.Shape(), bs, bs[len(bs)-2], oa.n)
		}
		bb, k = bs[:len(bs)-2], bs[len(bs)-1]
	}
	batch, ok := broadcast(oa.batch, bb)
	if !ok {
		return nil, fmt.Errorf("%w: Solve batch dimensions %v and %v do not broadcast", ErrShape, oa.batch, bb)
	}
	cplx, single := precision(a, b)
	shape := shapeOf(batch, oa.n)
	if !vec {
		shape = append(shape, k)
	}
	if cplx {
		return solveT[complex128](oa, b, batch, bb, k, shape, single)
	}
	return solveT[float64](oa, b, batch, bb, k, shape, single)
}

func solveT[T scalar](oa operand, b *ndarray.Array, batch, bb []int, k int, shape []int, single bool) (*ndarray.Array, error) {
	n := oa.n
	ad, bd := load[T](oa.a), load[T](b)
	cnt := prod(batch)
	out := make([]T, cnt*n*k)
	var fe firstErr
	parallelFor(cnt, n*n*(n+k), func(i int) {
		ai := batchIndex(batch, oa.batch, i)
		lu := append([]T(nil), ad[ai*n*n:(ai+1)*n*n]...)
		piv := make([]int, n)
		if getrf(n, n, lu, n, piv) >= 0 {
			fe.set(i, ErrSingular)
			return
		}
		x := out[i*n*k : (i+1)*n*k]
		bi := batchIndex(batch, bb, i)
		copy(x, bd[bi*n*k:])
		getrs(n, k, lu, piv, x)
	})
	if fe.err != nil {
		return nil, fe.err
	}
	return wrap(out, shape, single), nil
}

// Inv is numpy.linalg.inv: the inverse of each square matrix of a
// (..., M, M). An exactly singular matrix gives ErrSingular.
func Inv(a *ndarray.Array) (*ndarray.Array, error) {
	o, err := square(a, "Inv")
	if err != nil {
		return nil, err
	}
	if o.cplx {
		return invT[complex128](o)
	}
	return invT[float64](o)
}

func invT[T scalar](o operand) (*ndarray.Array, error) {
	n := o.n
	d := load[T](o.a)
	out := make([]T, len(d))
	var fe firstErr
	parallelFor(o.count(), 2*n*n*n, func(i int) {
		lu := d[i*n*n : (i+1)*n*n]
		piv := make([]int, n)
		if getrf(n, n, lu, n, piv) >= 0 {
			fe.set(i, ErrSingular)
			return
		}
		x := out[i*n*n : (i+1)*n*n]
		for j := 0; j < n; j++ {
			x[j*n+j] = 1
		}
		getrs(n, n, lu, piv, x)
	})
	if fe.err != nil {
		return nil, fe.err
	}
	return wrap(out, shapeOf(o.batch, n, n), o.single), nil
}

// Det is numpy.linalg.det: the determinant of each square matrix of a
// (..., M, M), shaped (...). It is computed from the LU factors, with the
// product of the pivots accumulated as a mantissa and an exponent, so it
// neither overflows nor underflows before the final result does. A singular
// matrix has determinant 0, not an error.
func Det(a *ndarray.Array) (*ndarray.Array, error) {
	o, err := square(a, "Det")
	if err != nil {
		return nil, err
	}
	if o.cplx {
		return detT[complex128](o)
	}
	return detT[float64](o)
}

func detT[T scalar](o operand) (*ndarray.Array, error) {
	signs, mants, exps, _ := luDets[T](o)
	out := make([]T, len(signs))
	for i := range out {
		out[i] = rscale(signs[i], math.Ldexp(mants[i], exps[i]))
	}
	return wrap(out, o.batch, o.single), nil
}

// Slogdet is numpy.linalg.slogdet: the sign and the natural logarithm of the
// absolute value of the determinant of each matrix of a (..., M, M), so that
// det = sign·exp(logabsdet) without overflow. For a complex input sign is the
// complex unit det/|det|. A singular matrix has sign 0 and logabsdet -Inf.
// sign has a's (promoted) dtype, logabsdet the matching real dtype.
func Slogdet(a *ndarray.Array) (sign, logabsdet *ndarray.Array, err error) {
	o, err := square(a, "Slogdet")
	if err != nil {
		return nil, nil, err
	}
	if o.cplx {
		return slogdetT[complex128](o)
	}
	return slogdetT[float64](o)
}

func slogdetT[T scalar](o operand) (*ndarray.Array, *ndarray.Array, error) {
	signs, _, _, logs := luDets[T](o)
	return wrap(signs, o.batch, o.single), wrap(logs, o.batch, o.single), nil
}

// luDets factors every matrix of o and returns, for each, the sign (a unit,
// or 0 when singular), the determinant's modulus as mantissa·2^exponent,
// and its natural logarithm.
func luDets[T scalar](o operand) (signs []T, mants []float64, exps []int, logs []float64) {
	n, cnt := o.n, o.count()
	d := load[T](o.a)
	signs, mants, exps, logs = make([]T, cnt), make([]float64, cnt), make([]int, cnt), make([]float64, cnt)
	parallelFor(cnt, n*n*n, func(i int) {
		lu := d[i*n*n : (i+1)*n*n]
		piv := make([]int, n)
		getrf(n, n, lu, n, piv)
		sign, mant, exp, logd := T(1), 1.0, 0, 0.0
		for j := 0; j < n; j++ {
			if piv[j] != j {
				sign = -sign
			}
			p := lu[j*n+j]
			ap := abs(p)
			if ap == 0 {
				sign, mant, logd = 0, 0, math.Inf(-1)
				break
			}
			sign *= unit(p, ap)
			f, e := math.Frexp(ap)
			mant, exp = mant*f, exp+e
			f, e = math.Frexp(mant)
			mant, exp = f, exp+e
			logd += math.Log(ap)
		}
		signs[i], mants[i], exps[i], logs[i] = sign, mant, exp, logd
	})
	return signs, mants, exps, logs
}

// unit is p/|p| (ap = |p| > 0).
func unit[T scalar](p T, ap float64) T {
	if z, ok := any(p).(complex128); ok {
		u := complex(real(z)/ap, imag(z)/ap)
		return any(u / complex(cmplx.Abs(u), 0)).(T)
	}
	return fromReal[T](math.Copysign(1, any(p).(float64)))
}
