package linalg

import (
	"math"

	"github.com/go-ndarray/ndarray"
)

// gebrdBlock is the panel width of the blocked bidiagonalization, and
// gebrdMin the order below which it is not blocked.
var gebrdBlock, gebrdMin = 32, 96

// bidiag is a bidiagonal reduction B = Qᴴ·A·P: B's diagonal d and
// superdiagonal e, the left reflectors (stored in the reduced matrix below
// its diagonal, with scalars tauq) and the right ones (in pv, geqrf's
// layout on an (n-1)×(n-1) block, with scalars taup), Q = H0·H1···H(n-1),
// P = G0·G1···G(n-2).
type bidiag[T scalar] struct {
	d, e       []float64
	tauq, taup []T
	pv         []T
}

// gebrd reduces the m×n matrix a (m >= n) to real upper bidiagonal form by
// Householder reflections from the left and the right (LAPACK gebrd):
// panels of gebrdBlock rows and columns as labrd, the trailing matrix
// updated once per panel through the GEMM, the rest as gebd2. The right
// reflectors are kept only when wantP.
func gebrd[T scalar](m, n int, a []T, wantP bool) bidiag[T] {
	b := bidiag[T]{
		d: make([]float64, n), e: make([]float64, max(n-1, 0)),
		tauq: make([]T, n), taup: make([]T, max(n-1, 0)),
	}
	if wantP && n > 1 {
		b.pv = make([]T, (n-1)*(n-1))
	}
	i0 := 0
	if n >= gebrdMin {
		nb := gebrdBlock
		for ; i0+nb < n-nb; i0 += nb {
			labrd(m, n, i0, nb, a, &b)
		}
	}
	gebd2(m, n, i0, a, &b)
	return b
}

// gebd2 is the unblocked bidiagonalization (LAPACK gebd2) of rows and
// columns i0 onwards.
func gebd2[T scalar](m, n, i0 int, a []T, b *bidiag[T]) {
	v := make([]T, m)
	w := make([]T, n)
	for i := i0; i < n; i++ {
		var x []T
		if i+1 < m {
			x = a[(i+1)*n+i:]
		}
		beta, tq := larfg(m-i, a[i*n+i], x, n)
		b.d[i], b.tauq[i] = re(beta), tq
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
		b.e[i], b.taup[i] = re(beta), tp
		row[0] = 1
		larfR(m-i-1, n-i-1, row, tp, a[(i+1)*n+i+1:], n)
		b.storeP(n, i, row)
		for c := range row {
			row[c] = conj(row[c])
		}
		row[0] = beta
	}
}

// storeP records the right reflector u (u[0] = 1, at column i+1) of step i.
func (b *bidiag[T]) storeP(n, i int, u []T) {
	if b.pv != nil {
		for c := 1; c < len(u); c++ {
			b.pv[(i+c)*(n-1)+i] = u[c]
		}
	}
}

// labrd reduces the nb rows and columns i0..i0+nb-1 of the m×n a (LAPACK
// labrd), accumulating X and Y so that the trailing matrix becomes
// A - V·Yᴴ - X·Uᴴ (V and U the left and right reflectors' vectors), which
// is then applied by two GEMMs. All four are kept transposed — one vector
// per row — so the corrections are dot products and axpys along contiguous
// rows; A's own products are matrix-vector products along rows, of A for X
// and of a transposed copy of the trailing block for Y.
func labrd[T scalar](m, n, i0, nb int, a []T, b *bidiag[T]) {
	mr, nc := m-i0, n-i0
	vt, xt := make([]T, nb*mr), make([]T, nb*mr)
	yt, ut := make([]T, nb*nc), make([]T, nb*nc)
	at := make([]T, nc*mr) // the trailing block, transposed and conjugated
	for r := 0; r < mr; r++ {
		row := a[(i0+r)*n+i0 : (i0+r)*n+n]
		for c, x := range row {
			at[c*mr+r] = conj(x)
		}
	}
	acc := make([]T, max(mr, nc))
	t1, t2 := make([]T, nb), make([]T, nb)
	for j := 0; j < nb; j++ {
		i := i0 + j
		// Column i of the current matrix, rows i onwards.
		col := acc[:mr-j]
		clear(col)
		for p := 0; p < j; p++ {
			axpy(col, vt[p*mr+j:(p+1)*mr], conj(yt[p*nc+j]))
			axpy(col, xt[p*mr+j:(p+1)*mr], conj(ut[p*nc+j]))
		}
		for r, x := range col {
			a[(i+r)*n+i] -= x
		}
		var x []T
		if i+1 < m {
			x = a[(i+1)*n+i:]
		}
		beta, tq := larfg(mr-j, a[i*n+i], x, n)
		b.d[i], b.tauq[i] = re(beta), tq
		a[i*n+i] = beta
		v := vt[j*mr : (j+1)*mr]
		v[j] = 1
		for r := j + 1; r < mr; r++ {
			v[r] = a[(i0+r)*n+i]
		}
		// y = tauq·(Aᴴv - Y·(Vᴴv) - U·(Xᴴv)), entries j+1 onwards.
		y := yt[j*nc+j+1 : (j+1)*nc]
		vj := v[j:]
		matvec(nc-j-1, mr-j, at[(j+1)*mr+j:], mr, vj, y)
		for p := 0; p < j; p++ {
			t1[p] = dotc(vt[p*mr+j:(p+1)*mr], vj)
			t2[p] = dotc(xt[p*mr+j:(p+1)*mr], vj)
		}
		for p := 0; p < j; p++ {
			axpy(y, yt[p*nc+j+1:(p+1)*nc], -t1[p])
			axpy(y, ut[p*nc+j+1:(p+1)*nc], -t2[p])
		}
		for c := range y {
			y[c] *= tq
		}
		// Row i of the matrix after the left reflector, columns i+1 onwards.
		row := a[i*n+i+1 : i*n+n]
		rc := acc[:nc-j-1]
		clear(rc)
		for p := 0; p <= j; p++ {
			axpy(rc, yt[p*nc+j+1:(p+1)*nc], conj(vt[p*mr+j]))
		}
		for p := 0; p < j; p++ {
			axpy(rc, ut[p*nc+j+1:(p+1)*nc], conj(xt[p*mr+j]))
		}
		for c := range row {
			row[c] = conj(row[c] - conj(rc[c]))
		}
		var yy []T
		if i+2 < n {
			yy = row[1:]
		}
		beta, tp := larfg(nc-j-1, row[0], yy, 1)
		b.e[i], b.taup[i] = re(beta), tp
		row[0] = 1
		u := ut[j*nc+j+1 : (j+1)*nc]
		copy(u, row)
		b.storeP(n, i, row)
		for c := range row {
			row[c] = conj(row[c])
		}
		row[0] = beta
		// x = taup·(A·u - V·(Yᴴu) - X·(Uᴴu)), entries j+1 onwards.
		xr := xt[j*mr+j+1 : (j+1)*mr]
		matvec(mr-j-1, nc-j-1, a[(i+1)*n+i+1:], n, u, xr)
		for p := 0; p <= j; p++ {
			t1[p] = dotc(yt[p*nc+j+1:(p+1)*nc], u)
		}
		for p := 0; p < j; p++ {
			t2[p] = dotc(ut[p*nc+j+1:(p+1)*nc], u)
		}
		for p := 0; p <= j; p++ {
			axpy(xr, vt[p*mr+j+1:(p+1)*mr], -t1[p])
		}
		for p := 0; p < j; p++ {
			axpy(xr, xt[p*mr+j+1:(p+1)*mr], -t2[p])
		}
		for r := range xr {
			xr[r] *= tp
		}
	}
	// The trailing update.
	tm, tn := mr-nb, nc-nb
	c0 := (i0+nb)*n + i0 + nb
	vv, xx := rm(vt, nb, mr), rm(xt, nb, mr) // nb×tm: Vᵀ and Xᵀ
	yy, uu := rm(yt, nb, nc), rm(ut, nb, nc) // nb×tn: Yᵀ and Uᵀ
	gemm(tm, tn, nb, -1, vv.t(), yy.c(), 1, a[c0:], n)
	gemm(tm, tn, nb, -1, xx.t(), uu.c(), 1, a[c0:], n)
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
	bd := gebrd(m, n, x, wantUV)
	d, e, tauq, pv, taup := bd.d, bd.e, bd.tauq, bd.pv, bd.taup
	if !wantUV {
		return d, nil, nil, bdsqr[T](n, d, e, nil, 0, nil, 0)
	}
	ncu := n
	if full {
		ncu = m
	}
	q := orgqr(m, ncu, n, x, n, tauq)
	ph := make([]T, n*n) // Pᴴ
	if n > 0 {
		ph[0] = 1
	}
	if n > 1 {
		p := orgqr(n-1, n-1, n-1, pv, n-1, taup)
		for r := 0; r < n-1; r++ {
			for c := 0; c < n-1; c++ {
				ph[(c+1)*n+r+1] = conj(p[r*(n-1)+c])
			}
		}
	}
	if n > dcLeaf {
		// Divide and conquer on the bidiagonal, then U = Q·U_B, Vᴴ = V_Bᵀ·Pᴴ.
		ub, vtb, ok := bdsdc(n, d, e)
		if !ok {
			return nil, nil, nil, false
		}
		u := mulMixed(m, n, n, rm(q, 0, ncu), rm(ub, 0, n))
		if ncu > n {
			full := make([]T, m*ncu)
			for r := 0; r < m; r++ {
				copy(full[r*ncu:r*ncu+n], u[r*n:(r+1)*n])
				copy(full[r*ncu+n:(r+1)*ncu], q[r*ncu+n:(r+1)*ncu])
			}
			u = full
		}
		return d, u, mulMixedLeft(n, n, n, rm(vtb, 0, n), rm(ph, 0, n)), true
	}
	ut := transposed(m, ncu, q)
	if !bdsqr(n, d, e, ph, n, ut, m) {
		return nil, nil, nil, false
	}
	return d, transposed(ncu, m, ut), ph, true
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
		if wantUV {
			copy(us[i*m*mu:], u)
			copy(vs[i*nv*n:], vh)
		}
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
