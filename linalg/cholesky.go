package linalg

import (
	"math"

	"github.com/go-ndarray/ndarray"
)

// potrf factors the Hermitian positive definite n×n matrix whose lower
// triangle is at a (leading dimension lda) as L·Lᴴ, L overwriting the lower
// triangle (LAPACK potrf, recursive). The strictly upper triangle is used as
// scratch. It returns -1, or the index of the leading minor that is not
// positive definite.
func potrf[T scalar](n int, a []T, lda int) int {
	if n <= trsmLeaf {
		for j := 0; j < n; j++ {
			rj := a[j*lda : j*lda+j]
			d := re(a[j*lda+j])
			for _, v := range rj {
				d -= absSq(v)
			}
			if !(d > 0) { // also catches NaN
				return j
			}
			d = math.Sqrt(d)
			a[j*lda+j] = fromReal[T](d)
			for i := j + 1; i < n; i++ {
				ri := a[i*lda : i*lda+j]
				s := a[i*lda+j]
				for k, v := range rj {
					s -= ri[k] * conj(v)
				}
				a[i*lda+j] = rscale(s, 1/d)
			}
		}
		return -1
	}
	n1 := n / 2
	if info := potrf(n1, a, lda); info >= 0 {
		return info
	}
	trsmRLC(n-n1, n1, a, lda, a[n1*lda:], lda)
	herkLower(n-n1, n1, a[n1*lda:], lda, a[n1*lda+n1:], lda)
	if info := potrf(n-n1, a[n1*lda+n1:], lda); info >= 0 {
		return info + n1
	}
	return -1
}

// Cholesky is numpy.linalg.cholesky: for each Hermitian positive definite
// matrix A of a (..., M, M), the lower triangular L with A = L·Lᴴ (uplo
// Lower), or the upper triangular U with A = Uᴴ·U (uplo Upper, numpy's
// upper=True). Only that triangle of A is read. A matrix that is not
// positive definite gives ErrNotPosDef.
func Cholesky(a *ndarray.Array, uplo UPLO) (*ndarray.Array, error) {
	o, err := square(a, "Cholesky")
	if err != nil {
		return nil, err
	}
	if o.cplx {
		return choleskyT[complex128](o, uplo)
	}
	return choleskyT[float64](o, uplo)
}

func choleskyT[T scalar](o operand, uplo UPLO) (*ndarray.Array, error) {
	n := o.n
	d := load[T](o.a)
	var fe firstErr
	parallelFor(o.count(), n*n*n/3, func(i int) {
		x := d[i*n*n : (i+1)*n*n]
		if uplo == Upper {
			conjTranspose(n, x)
		}
		if potrf(n, x, n) >= 0 {
			fe.set(i, ErrNotPosDef)
			return
		}
		for r := 0; r < n; r++ {
			clear(x[r*n+r+1 : r*n+n])
		}
		if uplo == Upper {
			conjTranspose(n, x)
		}
	})
	if fe.err != nil {
		return nil, fe.err
	}
	return wrap(d, shapeOf(o.batch, n, n), o.single), nil
}

// conjTranspose replaces the n×n matrix x by its conjugate transpose.
func conjTranspose[T scalar](n int, x []T) {
	for r := 0; r < n; r++ {
		x[r*n+r] = conj(x[r*n+r])
		for c := r + 1; c < n; c++ {
			x[r*n+c], x[c*n+r] = conj(x[c*n+r]), conj(x[r*n+c])
		}
	}
}
