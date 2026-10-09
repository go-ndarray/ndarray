package linalg

import (
	"math"

	"github.com/go-ndarray/ndarray/internal/kernels"

	"github.com/go-ndarray/ndarray"
)

// hermitianFrom completes the n×n matrix a into a Hermitian one from the
// triangle uplo selects, dropping the imaginary part of the diagonal as
// LAPACK does.
func hermitianFrom[T scalar](n int, a []T, uplo UPLO) {
	for r := 0; r < n; r++ {
		a[r*n+r] = fromReal[T](re(a[r*n+r]))
		for c := r + 1; c < n; c++ {
			if uplo == Lower {
				a[r*n+c] = conj(a[c*n+r])
			} else {
				a[c*n+r] = conj(a[r*n+c])
			}
		}
	}
}

// hetrdBlock is the panel width of the blocked tridiagonalization, and
// hetrdMin the order below which it is not blocked.
var hetrdBlock, hetrdMin = 32, 96

// hetrd reduces the full Hermitian n×n matrix a to real symmetric
// tridiagonal form T = Qᴴ·A·Q by Householder reflections (LAPACK hetrd, on
// the lower triangle). Panels of hetrdBlock columns are reduced as in
// latrd, accumulating W so the trailing matrix is updated once per panel by
// A -= V·Wᴴ + W·Vᴴ through the GEMM; the rest is hetd2. The matrix is kept
// in full storage, so the matrix-vector products run along contiguous rows.
// It returns the diagonal d, the off-diagonal e and, when wantQ, Q as an
// n×n row-major matrix. a is destroyed.
func hetrd[T scalar](n int, a []T, wantQ bool) (d, e []float64, q []T) {
	d = make([]float64, n)
	e = make([]float64, max(n-1, 0))
	var refl []T // the reflectors, in geqrf's layout on the trailing (n-1)×(n-1) block
	taus := make([]T, max(n-1, 0))
	if wantQ && n > 1 {
		refl = make([]T, (n-1)*(n-1))
	}
	store := func(i int, v []T, tau T) {
		taus[i] = tau
		if refl != nil {
			for r := 1; r < len(v); r++ {
				refl[(i+r)*(n-1)+i] = v[r]
			}
		}
	}
	i := 0
	if n >= hetrdMin {
		nb := hetrdBlock
		for ; i+nb < n-nb; i += nb {
			latrd(n, i, nb, a, d, e, store)
		}
	}
	v := make([]T, n)
	p := make([]T, n)
	for ; i < n-1; i++ {
		s := n - i - 1 // order of the trailing block A22 = a[i+1:, i+1:]
		var x []T
		if s > 1 {
			x = a[(i+2)*n+i:]
		}
		beta, tau := larfg(s, a[(i+1)*n+i], x, n)
		e[i] = re(beta)
		v[0] = 1
		for r := 1; r < s; r++ {
			v[r] = a[(i+1+r)*n+i]
		}
		if tau != 0 {
			// p = tau·A22·v; w = p - (tau/2)(pᴴv)·v; A22 -= v·wᴴ + w·vᴴ.
			matvec(s, s, a[(i+1)*n+i+1:], n, v, p)
			var pv T
			for r := 0; r < s; r++ {
				p[r] *= tau
				pv += conj(p[r]) * v[r]
			}
			alpha := -0.5 * tau * pv
			for r := 0; r < s; r++ {
				p[r] += alpha * v[r]
			}
			for r := 0; r < s; r++ {
				row := a[(i+1+r)*n+i+1 : (i+1+r)*n+n]
				vr, wr := v[r], p[r]
				for c := range row {
					row[c] -= vr*conj(p[c]) + wr*conj(v[c])
				}
			}
		}
		d[i] = re(a[i*n+i])
		store(i, v[:s], tau)
	}
	if n > 0 {
		d[n-1] = re(a[(n-1)*n+n-1])
	}
	if !wantQ {
		return d, e, nil
	}
	q = make([]T, n*n)
	if n > 0 {
		q[0] = 1
	}
	if n > 1 {
		qs := orgqr(n-1, n-1, n-1, refl, n-1, taus)
		for r := 0; r < n-1; r++ {
			copy(q[(r+1)*n+1:(r+2)*n], qs[r*(n-1):(r+1)*(n-1)])
		}
	}
	return d, e, q
}

// latrd reduces the nb columns i0..i0+nb-1 of the full Hermitian n×n a
// (LAPACK latrd, lower), then updates the trailing matrix
// a[i0+nb:, i0+nb:] -= V·Wᴴ + W·Vᴴ with the panel's reflectors V and the
// accumulated W. V and W are kept transposed (one reflector per row), so
// the corrections are dot products and axpys along contiguous rows.
func latrd[T scalar](n, i0, nb int, a []T, d, e []float64, store func(int, []T, T)) {
	m := n - i0 // rows of the panel, indexed from i0
	vt := make([]T, nb*m)
	wt := make([]T, nb*m)
	vc := make([]T, m)
	tmp := make([]T, m)
	t1, t2 := make([]T, nb), make([]T, nb)
	for j := 0; j < nb; j++ {
		c := i0 + j
		// Bring column c up to date with the panel's earlier reflectors:
		// a[c:, c] -= V·conj(W[j, :j]) + W·conj(V[j, :j]).
		if j > 0 {
			acc := tmp[:m-j]
			clear(acc)
			for p := 0; p < j; p++ {
				vp, wp := vt[p*m:(p+1)*m], wt[p*m:(p+1)*m]
				axpy(acc, vp[j:], conj(wp[j]))
				axpy(acc, wp[j:], conj(vp[j]))
			}
			for r, x := range acc {
				a[(c+r)*n+c] -= x
			}
		}
		d[c] = re(a[c*n+c])
		s := n - c - 1
		var x []T
		if s > 1 {
			x = a[(c+2)*n+c:]
		}
		beta, tau := larfg(s, a[(c+1)*n+c], x, n)
		e[c] = re(beta)
		v := vc[:s] // the reflector's vector
		v[0] = 1
		for r := 1; r < s; r++ {
			v[r] = a[(c+1+r)*n+c]
		}
		copy(vt[j*m+j+1:], v)
		store(c, v, tau)
		// w = tau·(A22 - V·Wᴴ - W·Vᴴ)·v, then w -= (tau/2)(wᴴv)·v.
		w := wt[j*m+j+1 : (j+1)*m]
		matvec(s, s, a[(c+1)*n+c+1:], n, v, w)
		for p := 0; p < j; p++ {
			t1[p] = dotc(wt[p*m+j+1:(p+1)*m], v)
			t2[p] = dotc(vt[p*m+j+1:(p+1)*m], v)
		}
		for p := 0; p < j; p++ {
			axpy(w, vt[p*m+j+1:(p+1)*m], -t1[p])
			axpy(w, wt[p*m+j+1:(p+1)*m], -t2[p])
		}
		var pv T
		for r := range w {
			w[r] *= tau
			pv += conj(w[r]) * v[r]
		}
		axpy(w, v, -0.5*tau*pv)
	}
	// The trailing update, on the full block.
	t := m - nb
	c0 := (i0+nb)*n + i0 + nb
	vv, ww := rm(vt, nb, m), rm(wt, nb, m) // nb×t: Vᵀ and Wᵀ
	gemm(t, t, nb, -1, vv.t(), ww.c(), 1, a[c0:], n)
	gemm(t, t, nb, -1, ww.t(), vv.c(), 1, a[c0:], n)
}

// axpy is y += alpha·x over len(x) elements.
func axpy[T scalar](y, x []T, alpha T) {
	y = y[:len(x)]
	for i, v := range x {
		y[i] += alpha * v
	}
}

// dotc is xᴴ·y.
func dotc[T scalar](x, y []T) T {
	if xf, ok := any(x).([]float64); ok {
		return any(kernels.Dot1DP(xf, any(y).([]float64)[:len(xf)])).(T)
	}
	var s T
	for i, v := range x {
		s += conj(v) * y[i]
	}
	return s
}

// matvecPar is the size (m·k) from which matvec spreads its rows over the
// processors: below it, waking the workers costs more than the work.
var matvecPar = 1 << 18

// matvec computes y = A·x for the m×k row-major A (leading dimension lda),
// by contiguous row dot products — SIMD and spread over the processors for
// float64.
func matvec[T scalar](m, k int, a []T, lda int, x, y []T) {
	if af, ok := any(a).([]float64); ok {
		if m*k < matvecPar {
			yf, xf := any(y).([]float64), any(x).([]float64)
			for r := 0; r < m; r++ {
				yf[r] = kernels.Dot1DP(af[r*lda:r*lda+k], xf[:k])
			}
			return
		}
		kernels.MatVecStridedP(any(y).([]float64), af, lda, any(x).([]float64)[:k], m, k)
		return
	}
	for r := 0; r < m; r++ {
		row := a[r*lda : r*lda+k]
		var s T
		for c, v := range row {
			s += v * x[c]
		}
		y[r] = s
	}
}

// laev2 is the eigendecomposition of the symmetric 2×2 [a b; b c] (LAPACK
// dlaev2): rt1 has the larger absolute value, and (cs, sn) is the unit right
// eigenvector for rt1.
func laev2(a, b, c float64) (rt1, rt2, cs, sn float64) {
	sm, df := a+c, a-c
	adf, tb := math.Abs(df), b+b
	ab := math.Abs(tb)
	acmx, acmn := a, c
	if math.Abs(a) <= math.Abs(c) {
		acmx, acmn = c, a
	}
	var rt float64
	switch {
	case adf > ab:
		rt = adf * math.Sqrt(1+(ab/adf)*(ab/adf))
	case adf < ab:
		rt = ab * math.Sqrt(1+(adf/ab)*(adf/ab))
	default:
		rt = ab * math.Sqrt2
	}
	var sgn1 float64
	switch {
	case sm < 0:
		rt1, sgn1 = 0.5*(sm-rt), -1
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
	case sm > 0:
		rt1, sgn1 = 0.5*(sm+rt), 1
		rt2 = (acmx/rt1)*acmn - (b/rt1)*b
	default:
		rt1, rt2, sgn1 = 0.5*rt, -0.5*rt, 1
	}
	var cs0, sgn2 float64
	if df >= 0 {
		cs0, sgn2 = df+rt, 1
	} else {
		cs0, sgn2 = df-rt, -1
	}
	if math.Abs(cs0) > ab {
		ct := -tb / cs0
		sn = 1 / math.Sqrt(1+ct*ct)
		cs = ct * sn
	} else if ab == 0 {
		cs, sn = 1, 0
	} else {
		tn := -cs0 / tb
		cs = 1 / math.Sqrt(1+tn*tn)
		sn = tn * cs
	}
	if sgn1 == sgn2 {
		cs, sn = -sn, cs
	}
	return rt1, rt2, cs, sn
}

// steqr computes the eigenvalues of the symmetric tridiagonal matrix (d, e)
// by the implicit QL or QR method with Wilkinson shifts (LAPACK dsteqr),
// sorted ascending into d. When z is not nil, its n rows of length ldz are
// the columns of a matrix Z (z holds Zᵀ) and receive the same rotations,
// so Z·(eigenvectors of T) comes out — row k of z is the eigenvector of
// d[k]. It reports false if some eigenvalue did not converge in 30·n
// iterations.
func steqr[T scalar](n int, d, e []float64, z []T, ldz int) bool {
	if n <= 1 {
		return true
	}
	row := func(i int) []T { return z[i*ldz : i*ldz+ldz] }
	rotate := func(i, j int, c, s float64) {
		if z != nil {
			rot(row(i), row(j), c, s)
		}
	}
	const eps2 = eps * eps
	ssfmax := math.Sqrt(1/safmin) / 3
	ssfmin := math.Sqrt(safmin) / eps2
	nmaxit := n * 30
	jtot := 0
	for l1 := 0; l1 < n; {
		if l1 > 0 {
			e[l1-1] = 0
		}
		m := n - 1
		for mm := l1; mm < n-1; mm++ {
			tst := math.Abs(e[mm])
			if tst == 0 {
				m = mm
				break
			}
			if tst <= math.Sqrt(math.Abs(d[mm]))*math.Sqrt(math.Abs(d[mm+1]))*eps {
				e[mm] = 0
				m = mm
				break
			}
		}
		l, lend := l1, m
		lsv, lendsv := l, lend
		l1 = m + 1
		if lend == l {
			continue
		}
		// Scale the submatrix in rows and columns l to lend.
		anorm := 0.0
		for i := l; i <= lend; i++ {
			anorm = math.Max(anorm, math.Abs(d[i]))
			if i < lend {
				anorm = math.Max(anorm, math.Abs(e[i]))
			}
		}
		if math.IsNaN(anorm) {
			return false
		}
		scale := 1.0
		switch {
		case anorm == 0:
			continue
		case anorm > ssfmax:
			scale = ssfmax / anorm
		case anorm < ssfmin:
			scale = ssfmin / anorm
		}
		scaleTri(d, e, lsv, lendsv, scale)
		// Choose between QL and QR iteration.
		if math.Abs(d[lend]) < math.Abs(d[l]) {
			l, lend = lendsv, lsv
		}
		if lend > l {
			// QL iteration: look for a small subdiagonal element.
			for l <= lend {
				m = lend
				for mm := l; mm < lend; mm++ {
					if tst := e[mm] * e[mm]; tst <= (eps2*math.Abs(d[mm]))*math.Abs(d[mm+1])+safmin {
						m = mm
						break
					}
				}
				if m < lend {
					e[m] = 0
				}
				p := d[l]
				if m == l {
					l++ // eigenvalue found
					continue
				}
				if m == l+1 {
					rt1, rt2, c, s := laev2(d[l], e[l], d[l+1])
					rotate(l, l+1, c, s)
					d[l], d[l+1], e[l] = rt1, rt2, 0
					l += 2
					continue
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				g := (d[l+1] - p) / (2 * e[l])
				r := lapy2(g, 1)
				g = d[m] - p + (e[l] / (g + math.Copysign(r, g)))
				s, c := 1.0, 1.0
				p = 0
				for i := m - 1; i >= l; i-- {
					f, b := s*e[i], c*e[i]
					c, s, r = lartg(g, f)
					if i != m-1 {
						e[i+1] = r
					}
					g = d[i+1] - p
					r = (d[i]-g)*s + 2*c*b
					p = s * r
					d[i+1] = g + p
					g = c*r - b
					rotate(i, i+1, c, -s)
				}
				d[l] -= p
				e[l] = g
			}
		} else {
			// QR iteration: look for a small superdiagonal element.
			for l >= lend {
				m = lend
				for mm := l; mm > lend; mm-- {
					if tst := e[mm-1] * e[mm-1]; tst <= (eps2*math.Abs(d[mm]))*math.Abs(d[mm-1])+safmin {
						m = mm
						break
					}
				}
				if m > lend {
					e[m-1] = 0
				}
				p := d[l]
				if m == l {
					l--
					continue
				}
				if m == l-1 {
					rt1, rt2, c, s := laev2(d[l-1], e[l-1], d[l])
					rotate(l-1, l, c, s)
					d[l-1], d[l], e[l-1] = rt1, rt2, 0
					l -= 2
					continue
				}
				if jtot == nmaxit {
					break
				}
				jtot++
				g := (d[l-1] - p) / (2 * e[l-1])
				r := lapy2(g, 1)
				g = d[m] - p + (e[l-1] / (g + math.Copysign(r, g)))
				s, c := 1.0, 1.0
				p = 0
				for i := m; i < l; i++ {
					f, b := s*e[i], c*e[i]
					c, s, r = lartg(g, f)
					if i != m {
						e[i-1] = r
					}
					g = d[i] - p
					r = (d[i+1]-g)*s + 2*c*b
					p = s * r
					d[i] = g + p
					g = c*r - b
					rotate(i, i+1, c, s)
				}
				d[l] -= p
				e[l-1] = g
			}
		}
		scaleTri(d, e, lsv, lendsv, 1/scale)
		if jtot >= nmaxit {
			for _, v := range e {
				if v != 0 {
					return false
				}
			}
			break
		}
	}
	// Sort the eigenvalues increasing, with their vectors (selection sort:
	// at most n-1 row swaps).
	for i := 0; i < n-1; i++ {
		k := i
		for j := i + 1; j < n; j++ {
			if d[j] < d[k] {
				k = j
			}
		}
		if k != i {
			d[i], d[k] = d[k], d[i]
			if z != nil {
				ri, rk := row(i), row(k)
				for c := range ri {
					ri[c], rk[c] = rk[c], ri[c]
				}
			}
		}
	}
	return true
}

// scaleTri multiplies d[lo..hi] and e[lo..hi-1] by s.
func scaleTri(d, e []float64, lo, hi int, s float64) {
	if s == 1 {
		return
	}
	for i := lo; i <= hi; i++ {
		d[i] *= s
		if i < hi {
			e[i] *= s
		}
	}
}

// Eigh is numpy.linalg.eigh: the eigenvalues, ascending, and the orthonormal
// eigenvectors (as columns) of each Hermitian matrix of a (..., M, M), read
// from the triangle uplo selects. w is (..., M) and real (float32 for
// single-precision inputs); v is (..., M, M) in a's precision. Eigenvectors
// are unique only up to a unit factor (a sign, or a phase), and within an
// eigenspace of a repeated eigenvalue only up to a unitary mix: compare
// A·v = λ·v, not the vectors themselves.
func Eigh(a *ndarray.Array, uplo UPLO) (w, v *ndarray.Array, err error) {
	o, err := square(a, "Eigh")
	if err != nil {
		return nil, nil, err
	}
	if o.cplx {
		return eighT[complex128](o, uplo, true)
	}
	return eighT[float64](o, uplo, true)
}

// Eigvalsh is numpy.linalg.eigvalsh: the eigenvalues of Eigh only, which
// costs a fraction of the vectors.
func Eigvalsh(a *ndarray.Array, uplo UPLO) (*ndarray.Array, error) {
	o, err := square(a, "Eigvalsh")
	if err != nil {
		return nil, err
	}
	var w *ndarray.Array
	if o.cplx {
		w, _, err = eighT[complex128](o, uplo, false)
	} else {
		w, _, err = eighT[float64](o, uplo, false)
	}
	return w, err
}

func eighT[T scalar](o operand, uplo UPLO, wantV bool) (*ndarray.Array, *ndarray.Array, error) {
	n, cnt := o.n, o.count()
	data := load[T](o.a)
	ws := make([]float64, cnt*n)
	var fe firstErr
	parallelFor(cnt, 4*n*n*n, func(i int) {
		x := data[i*n*n : (i+1)*n*n]
		vals, vecs, ok := eighOne(n, x, uplo, wantV)
		if !ok {
			fe.set(i, ErrNoConvergence)
			return
		}
		copy(ws[i*n:], vals)
		copy(x, vecs)
	})
	if fe.err != nil {
		return nil, nil, fe.err
	}
	w := wrap(ws, shapeOf(o.batch, n), o.single)
	if !wantV {
		return w, nil, nil
	}
	return w, wrap(data, shapeOf(o.batch, n, n), o.single), nil
}

// eighOne is the eigendecomposition of one Hermitian matrix: ascending
// eigenvalues and, when wantV, the eigenvectors as the columns of an n×n
// row-major matrix.
func eighOne[T scalar](n int, x []T, uplo UPLO, wantV bool) ([]float64, []T, bool) {
	hermitianFrom(n, x, uplo)
	d, e, q := hetrd(n, x, wantV)
	if !wantV {
		return d, nil, steqr[T](n, d, e, nil, 0)
	}
	if n > dcLeaf {
		zt, ok := stedc(n, d, e)
		if !ok {
			return nil, nil, false
		}
		return d, mulRealT(n, q, zt), true
	}
	zt := transposed(n, n, q) // rows of zt are the columns of Q
	if !steqr(n, d, e, zt, n) {
		return nil, nil, false
	}
	return d, transposed(n, n, zt), true
}

// mulRealT returns Q·Zᵀ for an n×n Q and a real n×n Z given as zt.
func mulRealT[T scalar](n int, q []T, zt []float64) []T {
	out := make([]T, n*n)
	if qf, ok := any(q).([]float64); ok {
		gemm(n, n, n, 1, rm(qf, 0, n), rm(zt, 0, n).t(), 0, any(out).([]float64), n)
		return out
	}
	qr, qi := split(n, n, rm(q, 0, n))
	re, im := make([]float64, n*n), make([]float64, n*n)
	gemm(n, n, n, 1, rm(qr, 0, n), rm(zt, 0, n).t(), 0, re, n)
	gemm(n, n, n, 1, rm(qi, 0, n), rm(zt, 0, n).t(), 0, im, n)
	oz := any(out).([]complex128)
	for i := range oz {
		oz[i] = complex(re[i], im[i])
	}
	return out
}

// transposed returns the n×m transpose of the m×n row-major matrix x.
func transposed[T scalar](m, n int, x []T) []T {
	out := make([]T, m*n)
	for r := 0; r < m; r++ {
		for c := 0; c < n; c++ {
			out[c*m+r] = x[r*n+c]
		}
	}
	return out
}
