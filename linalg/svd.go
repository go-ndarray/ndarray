package linalg

import (
	"math"

	"github.com/go-ndarray/ndarray"
)

// gebd2 reduces the m×n matrix a (m >= n) to real upper bidiagonal form
// B = Qᴴ·A·P by Householder reflections from the left and the right (LAPACK
// gebd2): Q = H0·H1···H(n-1), P = G0·G1···G(n-2). It returns B's diagonal d
// and superdiagonal e, the left reflectors (in a, below the diagonal, with
// scalars tauq) and the right ones (in pv, geqrf's layout on an
// (n-1)×(n-1) block, with scalars taup).
func gebd2[T scalar](m, n int, a []T, wantP bool) (d, e []float64, tauq, pv, taup []T) {
	d, e = make([]float64, n), make([]float64, max(n-1, 0))
	tauq, taup = make([]T, n), make([]T, max(n-1, 0))
	if wantP && n > 1 {
		pv = make([]T, (n-1)*(n-1))
	}
	v := make([]T, m)
	w := make([]T, n)
	for i := 0; i < n; i++ {
		var x []T
		if i+1 < m {
			x = a[(i+1)*n+i:]
		}
		beta, tq := larfg(m-i, a[i*n+i], x, n)
		d[i], tauq[i] = re(beta), tq
		if i+1 < n {
			v[0] = 1
			for r := i + 1; r < m; r++ {
				v[r-i] = a[r*n+i]
			}
			larfL(m-i, n-i-1, v[:m-i], conj(tq), a[i*n+i+1:], n, w)
		}
		a[i*n+i] = beta
		if i+1 >= n {
			continue
		}
		row := a[i*n+i+1 : i*n+n]
		for c := range row {
			row[c] = conj(row[c])
		}
		var y []T
		if i+2 < n {
			y = row[1:]
		}
		beta, tp := larfg(n-i-1, row[0], y, 1)
		e[i], taup[i] = re(beta), tp
		row[0] = 1
		larfR(m-i-1, n-i-1, row, tp, a[(i+1)*n+i+1:], n)
		if pv != nil {
			for c := 1; c < len(row); c++ {
				pv[(i+c)*(n-1)+i] = row[c]
			}
		}
		for c := range row {
			row[c] = conj(row[c])
		}
		row[0] = beta
	}
	return d, e, tauq, pv, taup
}

// las2 returns the singular values of the 2×2 upper triangular [f g; 0 h]
// (LAPACK dlas2).
func las2(f, g, h float64) (ssmin, ssmax float64) {
	fa, ga, ha := math.Abs(f), math.Abs(g), math.Abs(h)
	fhmn, fhmx := math.Min(fa, ha), math.Max(fa, ha)
	switch {
	case fhmn == 0:
		if fhmx == 0 {
			return 0, ga
		}
		mx, mn := math.Max(fhmx, ga), math.Min(fhmx, ga)
		return 0, mx * math.Sqrt(1+(mn/mx)*(mn/mx))
	case ga < fhmx:
		as := 1 + fhmn/fhmx
		at := (fhmx - fhmn) / fhmx
		au := (ga / fhmx) * (ga / fhmx)
		c := 2 / (math.Sqrt(as*as+au) + math.Sqrt(at*at+au))
		return fhmn * c, fhmx / c
	}
	au := fhmx / ga
	if au == 0 {
		// Avoid underflow in the computation of ssmin.
		return (fhmn * fhmx) / ga, ga
	}
	as := 1 + fhmn/fhmx
	at := (fhmx - fhmn) / fhmx
	c := 1 / (math.Sqrt(1+(as*au)*(as*au)) + math.Sqrt(1+(at*au)*(at*au)))
	return 2 * (fhmn * c) * au, ga / (c + c)
}

// lasv2 is the SVD of the 2×2 upper triangular [f g; 0 h] (LAPACK dlasv2):
// [csl snl; -snl csl]·[f g; 0 h]·[csr -snr; snr csr] = [ssmax 0; 0 ssmin],
// with |ssmax| >= |ssmin|.
func lasv2(f, g, h float64) (ssmin, ssmax, snr, csr, snl, csl float64) {
	ft, fa, ht, ha := f, math.Abs(f), h, math.Abs(h)
	pmax := 1
	swap := ha > fa
	if swap {
		pmax = 3
		ft, ht = ht, ft
		fa, ha = ha, fa
	}
	gt, ga := g, math.Abs(g)
	var clt, crt, slt, srt float64
	if ga == 0 {
		ssmin, ssmax = ha, fa
		clt, crt, slt, srt = 1, 1, 0, 0
	} else {
		gasmal := true
		if ga > fa {
			pmax = 2
			if fa/ga < eps {
				gasmal = false
				ssmax = ga
				if ha > 1 {
					ssmin = fa / (ga / ha)
				} else {
					ssmin = (fa / ga) * ha
				}
				clt, slt = 1, ht/gt
				srt, crt = 1, ft/gt
			}
		}
		if gasmal {
			dd := fa - ha
			l := 1.0
			if dd != fa {
				l = dd / fa
			}
			m := gt / ft
			t := 2 - l
			mm, tt := m*m, t*t
			s := math.Sqrt(tt + mm)
			r := math.Abs(m)
			if l != 0 {
				r = math.Sqrt(l*l + mm)
			}
			a := 0.5 * (s + r)
			ssmin, ssmax = ha/a, fa*a
			switch {
			case mm == 0 && l == 0:
				t = math.Copysign(2, ft) * math.Copysign(1, gt)
			case mm == 0:
				t = gt/math.Copysign(dd, ft) + m/t
			default:
				t = (m/(s+t) + m/(r+l)) * (1 + a)
			}
			l = math.Sqrt(t*t + 4)
			crt, srt = 2/l, t/l
			clt = (crt + srt*m) / a
			slt = (ht / ft) * srt / a
		}
	}
	if swap {
		csl, snl, csr, snr = srt, crt, slt, clt
	} else {
		csl, snl, csr, snr = clt, slt, crt, srt
	}
	var tsign float64
	switch pmax {
	case 1:
		tsign = math.Copysign(1, csr) * math.Copysign(1, csl) * math.Copysign(1, f)
	case 2:
		tsign = math.Copysign(1, snr) * math.Copysign(1, csl) * math.Copysign(1, g)
	default:
		tsign = math.Copysign(1, snr) * math.Copysign(1, snl) * math.Copysign(1, h)
	}
	ssmax = math.Copysign(ssmax, tsign)
	ssmin = math.Copysign(ssmin, tsign*math.Copysign(1, f)*math.Copysign(1, h))
	return ssmin, ssmax, snr, csr, snl, csl
}

// bdsqr computes the SVD of the n×n real upper bidiagonal (d, e) by implicit
// zero-shift and shifted QR (LAPACK dbdsqr, Demmel–Kahan): singular values
// to high relative accuracy, sorted decreasing into d. vt (n rows of length
// ldvt) is premultiplied by the right singular vectors' transpose; ut holds
// the transpose of U (its rows are U's columns, ldu long, at least n of
// them) and is postmultiplied. Either may be nil. It reports false if the
// iteration did not converge.
func bdsqr[T scalar](n int, d, e []float64, vt []T, ldvt int, ut []T, ldu int) bool {
	rotV := func(i, j int, c, s float64) {
		if vt != nil {
			rot(vt[i*ldvt:i*ldvt+ldvt], vt[j*ldvt:j*ldvt+ldvt], c, s)
		}
	}
	rotU := func(i, j int, c, s float64) {
		if ut != nil {
			rot(ut[i*ldu:i*ldu+ldu], ut[j*ldu:j*ldu+ldu], c, s)
		}
	}
	const maxitr = 6
	const hndrth = 0.01
	if n > 1 {
		tolmul := math.Max(10, math.Min(100, math.Pow(eps, -0.125)))
		tol := tolmul * eps
		// Approximate the largest and the smallest singular values.
		smax := 0.0
		for i := 0; i < n; i++ {
			smax = math.Max(smax, math.Abs(d[i]))
		}
		for i := 0; i < n-1; i++ {
			smax = math.Max(smax, math.Abs(e[i]))
		}
		if math.IsNaN(smax) {
			return false
		}
		sminoa := math.Abs(d[0])
		if sminoa != 0 {
			mu := sminoa
			for i := 1; i < n && sminoa != 0; i++ {
				mu = math.Abs(d[i]) * (mu / (mu + math.Abs(e[i-1])))
				sminoa = math.Min(sminoa, mu)
			}
		}
		sminoa /= math.Sqrt(float64(n))
		thresh := math.Max(tol*sminoa, maxitr*(float64(n)*(float64(n)*safmin)))
		maxit := maxitr * n * n
		iter := 0
		oldll, oldm := -1, -1
		idir := 0
		m := n - 1 // the last element of the unconverged part
		for m > 0 {
			if iter > maxit {
				return false
			}
			// Find the diagonal block of the matrix to work on.
			smax = math.Abs(d[m])
			ll := -1
			for k := m - 1; k >= 0; k-- {
				abss, abse := math.Abs(d[k]), math.Abs(e[k])
				if abse <= thresh {
					ll = k
					break
				}
				smax = math.Max(smax, math.Max(abss, abse))
			}
			if ll >= 0 {
				e[ll] = 0
				if ll == m-1 {
					m-- // the bottom singular value converged
					continue
				}
			}
			ll++
			// e[ll..m-1] are nonzero, e[ll-1] is zero.
			if ll == m-1 {
				// A 2×2 block: handle it separately.
				sigmn, sigmx, sinr, cosr, sinl, cosl := lasv2(d[m-1], e[m-1], d[m])
				d[m-1], e[m-1], d[m] = sigmx, 0, sigmn
				rotV(m-1, m, cosr, sinr)
				rotU(m-1, m, cosl, sinl)
				m -= 2
				continue
			}
			// On a new submatrix, choose the direction of the bulge chase
			// from the end with the larger diagonal entry.
			if ll > oldm || m < oldll {
				if math.Abs(d[ll]) >= math.Abs(d[m]) {
					idir = 1 // top to bottom
				} else {
					idir = 2 // bottom to top
				}
			}
			// Convergence tests.
			var sminl float64
			converged := false
			if idir == 1 {
				if math.Abs(e[m-1]) <= tol*math.Abs(d[m]) {
					e[m-1] = 0
					continue
				}
				mu := math.Abs(d[ll])
				sminl = mu
				for k := ll; k < m; k++ {
					if math.Abs(e[k]) <= tol*mu {
						e[k] = 0
						converged = true
						break
					}
					mu = math.Abs(d[k+1]) * (mu / (mu + math.Abs(e[k])))
					sminl = math.Min(sminl, mu)
				}
			} else {
				if math.Abs(e[ll]) <= tol*math.Abs(d[ll]) {
					e[ll] = 0
					continue
				}
				mu := math.Abs(d[m])
				sminl = mu
				for k := m - 1; k >= ll; k-- {
					if math.Abs(e[k]) <= tol*mu {
						e[k] = 0
						converged = true
						break
					}
					mu = math.Abs(d[k]) * (mu / (mu + math.Abs(e[k])))
					sminl = math.Min(sminl, mu)
				}
			}
			if converged {
				continue
			}
			oldll, oldm = ll, m
			// The shift: zero if it would ruin relative accuracy.
			shift := 0.0
			if float64(n)*tol*(sminl/smax) > math.Max(eps, hndrth*tol) {
				var sll float64
				if idir == 1 {
					sll = math.Abs(d[ll])
					shift, _ = las2(d[m-1], e[m-1], d[m])
				} else {
					sll = math.Abs(d[m])
					shift, _ = las2(d[ll], e[ll], d[ll+1])
				}
				if sll > 0 && (shift/sll)*(shift/sll) < eps {
					shift = 0
				}
			}
			iter += m - ll
			switch {
			case shift == 0 && idir == 1:
				cs, oldcs := 1.0, 1.0
				var sn, oldsn, r float64
				for i := ll; i < m; i++ {
					cs, sn, r = lartg(d[i]*cs, e[i])
					if i > ll {
						e[i-1] = oldsn * r
					}
					oldcs, oldsn, d[i] = lartg(oldcs*r, d[i+1]*sn)
					rotV(i, i+1, cs, sn)
					rotU(i, i+1, oldcs, oldsn)
				}
				h := d[m] * cs
				d[m], e[m-1] = h*oldcs, h*oldsn
				if math.Abs(e[m-1]) <= thresh {
					e[m-1] = 0
				}
			case shift == 0:
				cs, oldcs := 1.0, 1.0
				var sn, oldsn, r float64
				for i := m; i > ll; i-- {
					cs, sn, r = lartg(d[i]*cs, e[i-1])
					if i < m {
						e[i] = oldsn * r
					}
					oldcs, oldsn, d[i] = lartg(oldcs*r, d[i-1]*sn)
					rotV(i-1, i, oldcs, -oldsn)
					rotU(i-1, i, cs, -sn)
				}
				h := d[ll] * cs
				d[ll], e[ll] = h*oldcs, h*oldsn
				if math.Abs(e[ll]) <= thresh {
					e[ll] = 0
				}
			case idir == 1:
				f := (math.Abs(d[ll]) - shift) * (math.Copysign(1, d[ll]) + shift/d[ll])
				g := e[ll]
				for i := ll; i < m; i++ {
					cosr, sinr, r := lartg(f, g)
					if i > ll {
						e[i-1] = r
					}
					f = cosr*d[i] + sinr*e[i]
					e[i] = cosr*e[i] - sinr*d[i]
					g = sinr * d[i+1]
					d[i+1] *= cosr
					cosl, sinl, r := lartg(f, g)
					d[i] = r
					f = cosl*e[i] + sinl*d[i+1]
					d[i+1] = cosl*d[i+1] - sinl*e[i]
					if i < m-1 {
						g = sinl * e[i+1]
						e[i+1] *= cosl
					}
					rotV(i, i+1, cosr, sinr)
					rotU(i, i+1, cosl, sinl)
				}
				e[m-1] = f
				if math.Abs(e[m-1]) <= thresh {
					e[m-1] = 0
				}
			default:
				f := (math.Abs(d[m]) - shift) * (math.Copysign(1, d[m]) + shift/d[m])
				g := e[m-1]
				for i := m; i > ll; i-- {
					cosr, sinr, r := lartg(f, g)
					if i < m {
						e[i] = r
					}
					f = cosr*d[i] + sinr*e[i-1]
					e[i-1] = cosr*e[i-1] - sinr*d[i]
					g = sinr * d[i-1]
					d[i-1] *= cosr
					cosl, sinl, r := lartg(f, g)
					d[i] = r
					f = cosl*e[i-1] + sinl*d[i-1]
					d[i-1] = cosl*d[i-1] - sinl*e[i-1]
					if i > ll+1 {
						g = sinl * e[i-2]
						e[i-2] *= cosl
					}
					rotV(i-1, i, cosl, -sinl)
					rotU(i-1, i, cosr, -sinr)
				}
				e[ll] = f
				if math.Abs(e[ll]) <= thresh {
					e[ll] = 0
				}
			}
		}
	}
	// Make the singular values positive, then sort them decreasing
	// (selection sort: one swap per position).
	for i := 0; i < n; i++ {
		if d[i] < 0 {
			d[i] = -d[i]
			if vt != nil {
				row := vt[i*ldvt : i*ldvt+ldvt]
				for c := range row {
					row[c] = -row[c]
				}
			}
		}
	}
	for i := 0; i < n-1; i++ {
		k := i
		for j := i + 1; j < n; j++ {
			if d[j] > d[k] {
				k = j
			}
		}
		if k != i {
			d[i], d[k] = d[k], d[i]
			swapRows(vt, ldvt, i, k)
			swapRows(ut, ldu, i, k)
		}
	}
	return true
}

// swapRows swaps rows i and k of the row-major x (no-op for a nil x).
func swapRows[T scalar](x []T, ld, i, k int) {
	if x == nil {
		return
	}
	ri, rk := x[i*ld:i*ld+ld], x[k*ld:k*ld+ld]
	for c := range ri {
		ri[c], rk[c] = rk[c], ri[c]
	}
}

// svdOne is the SVD of one m×n matrix x (destroyed): the singular values,
// decreasing, and when wantUV, U (m×m if full, else m×k) and Vᴴ (n×n if
// full, else k×n), row-major.
func svdOne[T scalar](m, n int, x []T, wantUV, full bool) (s []float64, u, vh []T, ok bool) {
	if m < n {
		// Factor Aᴴ = U'·S·V'ᴴ, so A = V'·S·U'ᴴ.
		xh := transposed(m, n, x)
		for i := range xh {
			xh[i] = conj(xh[i])
		}
		s, u2, vh2, ok := svdOne(n, m, xh, wantUV, full)
		if !wantUV || !ok {
			return s, nil, nil, ok
		}
		ku := len(u2) / n // columns of U'
		return s, adjoint(m, m, vh2), adjoint(n, ku, u2), true
	}
	d, e, tauq, pv, taup := gebd2(m, n, x, wantUV)
	if !wantUV {
		return d, nil, nil, bdsqr[T](n, d, e, nil, 0, nil, 0)
	}
	ncu := n
	if full {
		ncu = m
	}
	ut := transposed(m, ncu, orgqr(m, ncu, n, x, n, tauq))
	vt := make([]T, n*n)
	if n > 0 {
		vt[0] = 1
	}
	if n > 1 {
		p := orgqr(n-1, n-1, n-1, pv, n-1, taup)
		for r := 0; r < n-1; r++ {
			for c := 0; c < n-1; c++ {
				vt[(c+1)*n+r+1] = conj(p[r*(n-1)+c]) // Pᴴ
			}
		}
	}
	if !bdsqr(n, d, e, vt, n, ut, m) {
		return nil, nil, nil, false
	}
	return d, transposed(ncu, m, ut), vt, true
}

// adjoint returns the n×m conjugate transpose of the m×n matrix x.
func adjoint[T scalar](m, n int, x []T) []T {
	out := transposed(m, n, x)
	for i := range out {
		out[i] = conj(out[i])
	}
	return out
}

// SVD is numpy.linalg.svd: A = U·diag(S)·Vh for each matrix of a
// (..., M, N), the singular values S (..., K), K = min(M, N), real and
// decreasing. With fullMatrices U is (..., M, M) and Vh (..., N, N), as
// numpy's default; otherwise (..., M, K) and (..., K, N). Singular vectors
// are unique only up to a unit factor per pair (and a unitary mix within
// repeated singular values). It is Golub–Kahan bidiagonalisation followed by
// implicit-shift QR (LAPACK gesvd's algorithm); numpy calls gesdd, whose
// divide and conquer agrees to rounding.
func SVD(a *ndarray.Array, fullMatrices bool) (u, s, vh *ndarray.Array, err error) {
	o, err := stacked(a, "SVD")
	if err != nil {
		return nil, nil, nil, err
	}
	if o.cplx {
		return svdT[complex128](o, true, fullMatrices)
	}
	return svdT[float64](o, true, fullMatrices)
}

// SVDVals is numpy.linalg.svdvals (svd with compute_uv=False): the singular
// values of each matrix of a, decreasing, without the vectors.
func SVDVals(a *ndarray.Array) (*ndarray.Array, error) {
	o, err := stacked(a, "SVDVals")
	if err != nil {
		return nil, err
	}
	var s *ndarray.Array
	if o.cplx {
		_, s, _, err = svdT[complex128](o, false, false)
	} else {
		_, s, _, err = svdT[float64](o, false, false)
	}
	return s, err
}

func svdT[T scalar](o operand, wantUV, full bool) (*ndarray.Array, *ndarray.Array, *ndarray.Array, error) {
	m, n, cnt := o.m, o.n, o.count()
	k := min(m, n)
	mu, nv := k, k
	if full {
		mu, nv = m, n
	}
	data := load[T](o.a)
	ss := make([]float64, cnt*k)
	var us, vs []T
	if wantUV {
		us, vs = make([]T, cnt*m*mu), make([]T, cnt*nv*n)
	}
	var fe firstErr
	parallelFor(cnt, 8*m*n*k, func(i int) {
		s, u, vh, ok := svdOne(m, n, data[i*m*n:(i+1)*m*n], wantUV, full)
		if !ok {
			fe.set(i, ErrNoConvergence)
			return
		}
		copy(ss[i*k:], s)
		copy(us[i*m*mu:], u)
		copy(vs[i*nv*n:], vh)
	})
	if fe.err != nil {
		return nil, nil, nil, fe.err
	}
	s := wrap(ss, shapeOf(o.batch, k), o.single)
	if !wantUV {
		return nil, s, nil, nil
	}
	return wrap(us, shapeOf(o.batch, m, mu), o.single), s, wrap(vs, shapeOf(o.batch, nv, n), o.single), nil
}
