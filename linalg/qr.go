package linalg

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
)

// qrBlock is the panel width of the blocked Householder QR: each panel is
// factored with geqr2 and applied to the rest of the matrix as one block
// reflector, through the GEMM.
var qrBlock = 32

// geqr2 is the unblocked Householder QR of the m×n matrix a (LAPACK geqr2):
// R overwrites the upper triangle, the reflectors' vectors (with an implicit
// unit first entry) the part below it, and tau their scalars.
func geqr2[T scalar](m, n int, a []T, lda int, tau []T) {
	k := min(m, n)
	v := make([]T, m)
	w := make([]T, n)
	for i := 0; i < k; i++ {
		var x []T
		if i+1 < m {
			x = a[(i+1)*lda+i:]
		}
		beta, t := larfg(m-i, a[i*lda+i], x, lda)
		tau[i] = t
		if i+1 < n {
			v[0] = 1
			for r := i + 1; r < m; r++ {
				v[r-i] = a[r*lda+i]
			}
			larfL(m-i, n-i-1, v[:m-i], conj(t), a[i*lda+i+1:], lda, w)
		}
		a[i*lda+i] = beta
	}
}

// geqrf is the blocked Householder QR (LAPACK geqrf), with geqr2's output.
func geqrf[T scalar](m, n int, a []T, lda int, tau []T) {
	k := min(m, n)
	nb := qrBlock
	if k <= nb {
		geqr2(m, n, a, lda, tau)
		return
	}
	for j := 0; j < k; j += nb {
		jb := min(nb, k-j)
		geqr2(m-j, jb, a[j*lda+j:], lda, tau[j:j+jb])
		if j+jb < n {
			br := newBlockReflector(m-j, jb, a[j*lda+j:], lda, tau[j:j+jb])
			br.applyLeft(n-j-jb, a[j*lda+j+jb:], lda, true)
		}
	}
}

// orgqr forms the first q columns of Q = H1·H2···Hk from geqrf's output
// (LAPACK orgqr), as an m×q row-major matrix. The reflectors are applied
// backwards a block at a time.
func orgqr[T scalar](m, q, k int, a []T, lda int, tau []T) []T {
	out := make([]T, m*q)
	for i := 0; i < min(m, q); i++ {
		out[i*q+i] = 1
	}
	nb := qrBlock
	for j := (k - 1) / nb * nb; k > 0 && j >= 0; j -= nb {
		jb := min(nb, k-j)
		br := newBlockReflector(m-j, jb, a[j*lda+j:], lda, tau[j:j+jb])
		br.applyLeft(q-j, out[j*q+j:], q, false)
	}
	return out
}

// QRMode selects what QR returns, as numpy.linalg.qr's mode does.
type QRMode uint8

// The modes. With M×N matrices and K = min(M, N):
const (
	// ModeReduced (numpy 'reduced', the default) returns Q (..., M, K) and
	// R (..., K, N).
	ModeReduced QRMode = iota
	// ModeComplete returns Q (..., M, M) and R (..., M, N).
	ModeComplete
	// ModeR returns only R (..., K, N); q is nil.
	ModeR
	// ModeRaw returns LAPACK's compact form, as numpy's 'raw': h
	// (..., N, M), the transpose of the factored matrix with R in its upper
	// part and the Householder vectors below, and tau (..., K).
	ModeRaw
)

// QR is numpy.linalg.qr: the factorization A = Q·R of each matrix of a
// (..., M, N), Q with orthonormal columns and R upper triangular, by
// Householder reflections as LAPACK's geqrf computes it (R's diagonal may be
// negative, as in numpy). See QRMode for what is returned.
func QR(a *ndarray.Array, mode QRMode) (q, r *ndarray.Array, err error) {
	o, err := stacked(a, "QR")
	if err != nil {
		return nil, nil, err
	}
	if mode > ModeRaw {
		return nil, nil, fmt.Errorf("%w: QR mode %d", ErrArgs, mode)
	}
	if o.cplx {
		q, r = qrT[complex128](o, mode)
	} else {
		q, r = qrT[float64](o, mode)
	}
	return q, r, nil
}

func qrT[T scalar](o operand, mode QRMode) (*ndarray.Array, *ndarray.Array) {
	m, n, cnt := o.m, o.n, o.count()
	k := min(m, n)
	mc := k
	if mode == ModeComplete {
		mc = m
	}
	d := load[T](o.a)
	taus := make([]T, cnt*k)
	var qs, rs []T
	if mode != ModeRaw {
		rs = make([]T, cnt*mc*n)
	}
	if mode == ModeReduced || mode == ModeComplete {
		qs = make([]T, cnt*m*mc)
	}
	parallelFor(cnt, 2*m*n*k, func(i int) {
		x := d[i*m*n : (i+1)*m*n]
		tau := taus[i*k : (i+1)*k]
		geqrf(m, n, x, n, tau)
		if rs != nil {
			r := rs[i*mc*n : (i+1)*mc*n]
			for row := 0; row < k; row++ {
				copy(r[row*n+row:row*n+n], x[row*n+row:row*n+n])
			}
		}
		if qs != nil {
			copy(qs[i*m*mc:], orgqr(m, mc, k, x, n, tau))
		}
	})
	switch mode {
	case ModeRaw:
		h := make([]T, len(d))
		for i := 0; i < cnt; i++ {
			for r := 0; r < m; r++ {
				for c := 0; c < n; c++ {
					h[i*m*n+c*m+r] = d[i*m*n+r*n+c]
				}
			}
		}
		return wrap(h, shapeOf(o.batch, n, m), o.single), wrap(taus, shapeOf(o.batch, k), o.single)
	case ModeR:
		return nil, wrap(rs, shapeOf(o.batch, mc, n), o.single)
	}
	return wrap(qs, shapeOf(o.batch, m, mc), o.single), wrap(rs, shapeOf(o.batch, mc, n), o.single)
}
