package linalg

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/go-ndarray/ndarray"
)

// errNonFinite is numpy's "Array must not contain infs or NaNs", raised by
// eig and eigvals before any work.
var errNonFinite = fmt.Errorf("%w: array must not contain infs or NaNs", ErrLinAlg)

// gebal balances the n×n matrix a in place (LAPACK gebal, job 'B', 3.12):
// rows and columns that isolate eigenvalues are permuted to the ends,
// leaving the block ilo..ihi, which is then scaled by powers of two so its
// rows and columns have comparable norms. perm and scale record the
// permutation and scaling for gebak.
func gebal[T scalar](n int, a []T) (ilo, ihi int, perm []int, scale []float64) {
	perm, scale = make([]int, n), make([]float64, n)
	for i := range perm {
		perm[i], scale[i] = i, 1
	}
	at := func(i, j int) T { return a[i*n+j] }
	swapCols := func(i, j, rows int) {
		for r := 0; r < rows; r++ {
			a[r*n+i], a[r*n+j] = a[r*n+j], a[r*n+i]
		}
	}
	swapRowsFrom := func(i, j, c0 int) {
		for c := c0; c < n; c++ {
			a[i*n+c], a[j*n+c] = a[j*n+c], a[i*n+c]
		}
	}
	k, l := 0, n-1
	// Rows isolating an eigenvalue go to the bottom.
	for noconv := true; noconv; {
		noconv = false
		for i := l; i >= 0; i-- {
			canSwap := true
			for j := 0; j <= l; j++ {
				if i != j && at(i, j) != 0 {
					canSwap = false
					break
				}
			}
			if !canSwap {
				continue
			}
			perm[l] = i
			if i != l {
				swapCols(i, l, l+1)
				swapRowsFrom(i, l, k)
			}
			noconv = true
			if l == 0 {
				return 0, 0, perm, scale
			}
			l--
		}
	}
	// Columns isolating an eigenvalue go to the left.
	for noconv := true; noconv; {
		noconv = false
		for j := k; j <= l; j++ {
			canSwap := true
			for i := k; i <= l; i++ {
				if i != j && at(i, j) != 0 {
					canSwap = false
					break
				}
			}
			if !canSwap {
				continue
			}
			perm[k] = j
			if j != k {
				swapCols(j, k, l+1)
				swapRowsFrom(j, k, k)
			}
			noconv = true
			k++
		}
	}
	// Scale the block k..l.
	const radix, factor = 2.0, 0.95
	sfmin1 := safmin / ulp
	sfmax1 := 1 / sfmin1
	sfmin2 := sfmin1 * radix
	sfmax2 := 1 / sfmin2
	for noconv := true; noconv; {
		noconv = false
		for i := k; i <= l; i++ {
			c := nrm2(l-k+1, a[k*n+i:], n)
			r := nrm2(l-k+1, a[i*n+k:], 1)
			ca, ra := 0.0, 0.0
			bi, bv := 0, -1.0
			for row := 0; row <= l; row++ {
				if v := abs1(at(row, i)); v > bv {
					bi, bv = row, v
				}
			}
			ca = abs(at(bi, i))
			bi, bv = k, -1.0
			for col := k; col < n; col++ {
				if v := abs1(at(i, col)); v > bv {
					bi, bv = col, v
				}
			}
			ra = abs(at(i, bi))
			if c == 0 || r == 0 {
				continue
			}
			g := r / radix
			f := 1.0
			s := c + r
			for c < g && math.Max(f, math.Max(c, ca)) < sfmax2 && math.Min(r, math.Min(g, ra)) > sfmin2 {
				f *= radix
				c *= radix
				ca *= radix
				r /= radix
				g /= radix
				ra /= radix
			}
			g = c / radix
			for g >= r && math.Max(r, ra) < sfmax2 && math.Min(math.Min(f, c), math.Min(g, ca)) > sfmin2 {
				f /= radix
				c /= radix
				g /= radix
				ca /= radix
				r *= radix
				ra *= radix
			}
			if c+r >= factor*s {
				continue
			}
			if f < 1 && scale[i] < 1 && f*scale[i] <= sfmin1 {
				continue
			}
			if f > 1 && scale[i] > 1 && scale[i] >= sfmax1/f {
				continue
			}
			scale[i] *= f
			noconv = true
			for col := k; col < n; col++ {
				a[i*n+col] = rscale(a[i*n+col], 1/f)
			}
			for row := 0; row <= l; row++ {
				a[row*n+i] = rscale(a[row*n+i], f)
			}
		}
	}
	return k, l, perm, scale
}

// gebak undoes gebal on the right eigenvectors, the columns of the n×n v.
func gebak[T scalar](n, ilo, ihi int, perm []int, scale []float64, v []T) {
	for i := ilo; i <= ihi; i++ {
		if s := scale[i]; s != 1 {
			row := v[i*n : i*n+n]
			for c := range row {
				row[c] = rscale(row[c], s)
			}
		}
	}
	for ii := 0; ii < n; ii++ {
		i := ii
		if i >= ilo && i <= ihi {
			continue
		}
		if i < ilo {
			i = ilo - 1 - ii
		}
		if k := perm[i]; k != i {
			swapRows(v, n, i, k)
		}
	}
}

// gehrd reduces rows and columns ilo..ihi of the n×n a to upper Hessenberg
// form H = Qᴴ·A·Q (LAPACK gehd2); with wantQ it returns Q, row-major. The
// entries below the subdiagonal are zeroed.
func gehrd[T scalar](n, ilo, ihi int, a []T, wantQ bool) []T {
	nh := ihi - ilo // the number of reflectors
	taus := make([]T, max(nh, 0))
	v := make([]T, n)
	w := make([]T, n)
	for i := ilo; i < ihi; i++ {
		cnt := ihi - i // rows i+1..ihi
		var x []T
		if cnt > 1 {
			x = a[(i+2)*n+i:]
		}
		beta, tau := larfg(cnt, a[(i+1)*n+i], x, n)
		taus[i-ilo] = tau
		v[0] = 1
		for r := 1; r < cnt; r++ {
			v[r] = a[(i+1+r)*n+i]
		}
		larfR(ihi+1, cnt, v, tau, a[i+1:], n)
		larfL(cnt, n-i-1, v[:cnt], conj(tau), a[(i+1)*n+i+1:], n, w)
		a[(i+1)*n+i] = beta
	}
	var q []T
	if wantQ {
		q = make([]T, n*n)
		for i := 0; i < n; i++ {
			q[i*n+i] = 1
		}
		if nh > 0 {
			refl := make([]T, nh*nh)
			for r := 1; r < nh; r++ {
				for c := 0; c < r; c++ {
					refl[r*nh+c] = a[(ilo+1+r)*n+ilo+c]
				}
			}
			qs := orgqr(nh, nh, nh, refl, nh, taus)
			for r := 0; r < nh; r++ {
				copy(q[(ilo+1+r)*n+ilo+1:], qs[r*nh:(r+1)*nh])
			}
		}
	}
	for r := 2; r < n; r++ {
		clear(a[r*n : r*n+r-1])
	}
	return q
}

// lanv2 is the Schur factorization of the real 2×2 [a b; c d] in standard
// form (LAPACK dlanv2): it returns the new entries (either c = 0, or a = d
// and b·c < 0 for a complex pair), the eigenvalues and the rotation.
func lanv2(a, b, c, d float64) (aa, bb, cc, dd, rt1r, rt1i, rt2r, rt2i, cs, sn float64) {
	const multpl = 4
	const safmn2 = 0x1p-485 // base**int(log(safmin/ulp)/log(base)/2)
	const safmx2 = 1 / safmn2
	switch {
	case c == 0:
		cs, sn = 1, 0
	case b == 0:
		// Swap rows and columns.
		cs, sn = 0, 1
		a, d = d, a
		b, c = -c, 0
	case a-d == 0 && math.Signbit(b) != math.Signbit(c):
		cs, sn = 1, 0
	default:
		temp := a - d
		p := 0.5 * temp
		bcmax := math.Max(math.Abs(b), math.Abs(c))
		bcmis := math.Min(math.Abs(b), math.Abs(c)) * math.Copysign(1, b) * math.Copysign(1, c)
		scale := math.Max(math.Abs(p), bcmax)
		z := (p/scale)*p + (bcmax/scale)*bcmis
		if z >= multpl*ulp {
			// Real eigenvalues: compute a and d.
			z = p + math.Copysign(math.Sqrt(scale)*math.Sqrt(z), p)
			a = d + z
			d -= (bcmax / z) * bcmis
			tau := lapy2(c, z)
			cs, sn = z/tau, c/tau
			b -= c
			c = 0
			break
		}
		// Complex eigenvalues, or real (almost) equal ones: make the
		// diagonal entries equal.
		sigma := b + c
		for count := 0; count < 20; count++ {
			scale = math.Max(math.Abs(temp), math.Abs(sigma))
			if scale >= safmx2 {
				sigma *= safmn2
				temp *= safmn2
				continue
			}
			if scale <= safmn2 {
				sigma *= safmx2
				temp *= safmx2
				continue
			}
			break
		}
		p = 0.5 * temp
		tau := lapy2(sigma, temp)
		cs = math.Sqrt(0.5 * (1 + math.Abs(sigma)/tau))
		sn = -(p / (tau * cs)) * math.Copysign(1, sigma)
		// [aa bb; cc dd] = [a b; c d]·[cs -sn; sn cs]
		aa0 := a*cs + b*sn
		bb0 := -a*sn + b*cs
		cc0 := c*cs + d*sn
		dd0 := -c*sn + d*cs
		// [a b; c d] = [cs sn; -sn cs]·[aa bb; cc dd]
		a = aa0*cs + cc0*sn
		b = bb0*cs + dd0*sn
		c = -aa0*sn + cc0*cs
		d = -bb0*sn + dd0*cs
		temp = 0.5 * (a + d)
		a, d = temp, temp
		if c != 0 && b != 0 && math.Signbit(b) == math.Signbit(c) {
			// Real eigenvalues: reduce to upper triangular form.
			sab := math.Sqrt(math.Abs(b))
			sac := math.Sqrt(math.Abs(c))
			p = math.Copysign(sab*sac, c)
			tau = 1 / math.Sqrt(math.Abs(b+c))
			a = temp + p
			d = temp - p
			b -= c
			c = 0
			cs1, sn1 := sab*tau, sac*tau
			cs, sn = cs*cs1-sn*sn1, cs*sn1+sn*cs1
		} else if c != 0 && b == 0 {
			b, c = -c, 0
			cs, sn = -sn, cs
		}
	}
	rt1r, rt2r = a, d
	if c != 0 {
		rt1i = math.Sqrt(math.Abs(b)) * math.Sqrt(math.Abs(c))
		rt2i = -rt1i
	}
	return a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn
}

// Shift strategy constants shared by the two QR iterations (LAPACK).
const (
	kexsh = 10
	dat1  = 3.0 / 4
	dat2  = -0.4375
)

// lahqr is the double-shift QR algorithm (LAPACK dlahqr) on the real upper
// Hessenberg n×n h, rows and columns ilo..ihi: eigenvalues into wr, wi; with
// wantt, h becomes the real Schur form T; with zt (the transpose of Z, n×n)
// not nil, Z is postmultiplied by the Schur vectors. It reports false if an
// eigenvalue did not converge.
func lahqr(wantt bool, n, ilo, ihi int, h, wr, wi, zt []float64) bool {
	if ilo == ihi {
		wr[ilo], wi[ilo] = h[ilo*n+ilo], 0
		return true
	}
	H := func(i, j int) float64 { return h[i*n+j] }
	nh := ihi - ilo + 1
	smlnum := safmin * (float64(nh) / ulp)
	i1, i2 := 0, n-1
	itmax := 30 * max(10, nh)
	kdefl := 0
	var v [3]float64
	for i := ihi; i >= ilo; {
		l := ilo
		converged := false
		for its := 0; its <= itmax; its++ {
			// Look for a single small subdiagonal element.
			k := i
			for ; k > l; k-- {
				if math.Abs(H(k, k-1)) <= smlnum {
					break
				}
				tst := math.Abs(H(k-1, k-1)) + math.Abs(H(k, k))
				if tst == 0 {
					if k-2 >= ilo {
						tst += math.Abs(H(k-1, k-2))
					}
					if k+1 <= ihi {
						tst += math.Abs(H(k+1, k))
					}
				}
				// Ahues & Tisseur's conservative deflation criterion.
				if math.Abs(H(k, k-1)) <= ulp*tst {
					ab := math.Max(math.Abs(H(k, k-1)), math.Abs(H(k-1, k)))
					ba := math.Min(math.Abs(H(k, k-1)), math.Abs(H(k-1, k)))
					aa := math.Max(math.Abs(H(k, k)), math.Abs(H(k-1, k-1)-H(k, k)))
					bb := math.Min(math.Abs(H(k, k)), math.Abs(H(k-1, k-1)-H(k, k)))
					s := aa + ab
					if ba*(ab/s) <= math.Max(smlnum, ulp*(bb*(aa/s))) {
						break
					}
				}
			}
			l = k
			if l > ilo {
				h[l*n+l-1] = 0
			}
			if l >= i-1 {
				converged = true
				break
			}
			kdefl++
			if !wantt {
				i1, i2 = l, i
			}
			var h11, h12, h21, h22 float64
			switch {
			case kdefl%(2*kexsh) == 0:
				s := math.Abs(H(i, i-1)) + math.Abs(H(i-1, i-2))
				h11 = dat1*s + H(i, i)
				h12, h21, h22 = dat2*s, s, h11
			case kdefl%kexsh == 0:
				s := math.Abs(H(l+1, l)) + math.Abs(H(l+2, l+1))
				h11 = dat1*s + H(l, l)
				h12, h21, h22 = dat2*s, s, h11
			default:
				h11, h21, h12, h22 = H(i-1, i-1), H(i, i-1), H(i-1, i), H(i, i)
			}
			var rt1r, rt1i, rt2r, rt2i float64
			if s := math.Abs(h11) + math.Abs(h12) + math.Abs(h21) + math.Abs(h22); s != 0 {
				h11 /= s
				h21 /= s
				h12 /= s
				h22 /= s
				tr := (h11 + h22) / 2
				det := (h11-tr)*(h22-tr) - h12*h21
				rtdisc := math.Sqrt(math.Abs(det))
				if det >= 0 {
					// A complex conjugate pair of shifts.
					rt1r, rt1i = tr*s, rtdisc*s
					rt2r, rt2i = rt1r, -rt1i
				} else {
					// Two real shifts: use only the one closer to h22.
					rt1r, rt2r = tr+rtdisc, tr-rtdisc
					if math.Abs(rt1r-h22) <= math.Abs(rt2r-h22) {
						rt1r *= s
						rt2r = rt1r
					} else {
						rt2r *= s
						rt1r = rt2r
					}
				}
			}
			// Look for two consecutive small subdiagonal elements.
			m := i - 2
			for ; ; m-- {
				h21s := H(m+1, m)
				s := math.Abs(H(m, m)-rt2r) + math.Abs(rt2i) + math.Abs(h21s)
				h21s = H(m+1, m) / s
				v[0] = h21s*H(m, m+1) + (H(m, m)-rt1r)*((H(m, m)-rt2r)/s) - rt1i*(rt2i/s)
				v[1] = h21s * (H(m, m) + H(m+1, m+1) - rt1r - rt2r)
				v[2] = h21s * H(m+2, m+1)
				s = math.Abs(v[0]) + math.Abs(v[1]) + math.Abs(v[2])
				v[0] /= s
				v[1] /= s
				v[2] /= s
				if m == l {
					break
				}
				h00 := math.Abs(H(m, m-1)) * (math.Abs(v[1]) + math.Abs(v[2]))
				h01 := math.Abs(v[0]) * (math.Abs(H(m-1, m-1)) + math.Abs(H(m, m)) + math.Abs(H(m+1, m+1)))
				if h00 <= ulp*h01 {
					break
				}
			}
			// The double-shift QR step.
			for k := m; k < i; k++ {
				nr := min(3, i-k+1)
				if k > m {
					for r := 0; r < nr; r++ {
						v[r] = H(k+r, k-1)
					}
				}
				var t1 float64
				v[0], t1 = larfg(nr, v[0], v[1:], 1)
				if k > m {
					h[k*n+k-1] = v[0]
					h[(k+1)*n+k-1] = 0
					if k < i-1 {
						h[(k+2)*n+k-1] = 0
					}
				} else if m > l {
					// Not h(k,k-1) = -h(k,k-1): that misbehaves when v[1]
					// and v[2] underflow.
					h[k*n+k-1] *= 1 - t1
				}
				v2 := v[1]
				t2 := t1 * v2
				if nr == 3 {
					v3 := v[2]
					t3 := t1 * v3
					r0, r1, r2 := h[k*n:k*n+n], h[(k+1)*n:(k+1)*n+n], h[(k+2)*n:(k+2)*n+n]
					for j := k; j <= i2; j++ {
						sum := r0[j] + v2*r1[j] + v3*r2[j]
						r0[j] -= sum * t1
						r1[j] -= sum * t2
						r2[j] -= sum * t3
					}
					for j := i1; j <= min(k+3, i); j++ {
						row := h[j*n : j*n+n]
						sum := row[k] + v2*row[k+1] + v3*row[k+2]
						row[k] -= sum * t1
						row[k+1] -= sum * t2
						row[k+2] -= sum * t3
					}
					if zt != nil {
						z0, z1, z2 := zt[k*n:k*n+n], zt[(k+1)*n:(k+1)*n+n], zt[(k+2)*n:(k+2)*n+n]
						for j := range z0 {
							sum := z0[j] + v2*z1[j] + v3*z2[j]
							z0[j] -= sum * t1
							z1[j] -= sum * t2
							z2[j] -= sum * t3
						}
					}
					continue
				}
				r0, r1 := h[k*n:k*n+n], h[(k+1)*n:(k+1)*n+n]
				for j := k; j <= i2; j++ {
					sum := r0[j] + v2*r1[j]
					r0[j] -= sum * t1
					r1[j] -= sum * t2
				}
				for j := i1; j <= i; j++ {
					row := h[j*n : j*n+n]
					sum := row[k] + v2*row[k+1]
					row[k] -= sum * t1
					row[k+1] -= sum * t2
				}
				if zt != nil {
					z0, z1 := zt[k*n:k*n+n], zt[(k+1)*n:(k+1)*n+n]
					for j := range z0 {
						sum := z0[j] + v2*z1[j]
						z0[j] -= sum * t1
						z1[j] -= sum * t2
					}
				}
			}
		}
		if !converged {
			return false
		}
		if l == i {
			// One eigenvalue converged.
			wr[i], wi[i] = H(i, i), 0
		} else {
			// A pair converged: standardize the 2×2 block.
			a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn := lanv2(H(i-1, i-1), H(i-1, i), H(i, i-1), H(i, i))
			h[(i-1)*n+i-1], h[(i-1)*n+i], h[i*n+i-1], h[i*n+i] = a, b, c, d
			wr[i-1], wi[i-1], wr[i], wi[i] = rt1r, rt1i, rt2r, rt2i
			if wantt {
				if i2 > i {
					rot(h[(i-1)*n+i+1:(i-1)*n+i2+1], h[i*n+i+1:i*n+i2+1], cs, sn)
				}
				for r := i1; r < i-1; r++ {
					x, y := h[r*n+i-1], h[r*n+i]
					h[r*n+i-1], h[r*n+i] = cs*x+sn*y, cs*y-sn*x
				}
			}
			if zt != nil {
				rot(zt[(i-1)*n:i*n], zt[i*n:(i+1)*n], cs, sn)
			}
		}
		kdefl = 0
		i = l - 1
	}
	return true
}

// zlahqr is the single-shift complex QR algorithm (LAPACK zlahqr) on the
// complex upper Hessenberg n×n h, rows and columns ilo..ihi: eigenvalues
// into w; with wantt, h becomes the triangular Schur form; zt (the
// transpose of Z) is postmultiplied by the Schur vectors when not nil.
func zlahqr(wantt bool, n, ilo, ihi int, h, w, zt []complex128) bool {
	if ilo == ihi {
		w[ilo] = h[ilo*n+ilo]
		return true
	}
	H := func(i, j int) complex128 { return h[i*n+j] }
	jlo, jhi := ilo, ihi
	if wantt {
		jlo, jhi = 0, n-1
	}
	scaleRow := func(r, c0, c1 int, s complex128) { // h[r, c0..c1-1] *= s
		for c := c0; c < c1; c++ {
			h[r*n+c] *= s
		}
	}
	scaleCol := func(c, r0, r1 int, s complex128) { // h[r0..r1-1, c] *= s
		for r := r0; r < r1; r++ {
			h[r*n+c] *= s
		}
	}
	scaleZ := func(c int, s complex128) {
		if zt != nil {
			row := zt[c*n : c*n+n]
			for j := range row {
				row[j] *= s
			}
		}
	}
	// Make the subdiagonal real.
	for i := ilo + 1; i <= ihi; i++ {
		if x := H(i, i-1); imag(x) != 0 {
			sc := x / complex(abs1(x), 0)
			sc = cmplx.Conj(sc) / complex(cmplx.Abs(sc), 0)
			h[i*n+i-1] = complex(cmplx.Abs(x), 0)
			scaleRow(i, i, jhi+1, sc)
			scaleCol(i, jlo, min(jhi, i+1)+1, cmplx.Conj(sc))
			scaleZ(i, cmplx.Conj(sc))
		}
	}
	nh := ihi - ilo + 1
	smlnum := safmin * (float64(nh) / ulp)
	i1, i2 := 0, n-1
	itmax := 30 * max(10, nh)
	kdefl := 0
	var v [2]complex128
	for i := ihi; i >= ilo; {
		l := ilo
		converged := false
		for its := 0; its <= itmax; its++ {
			k := i
			for ; k > l; k-- {
				if abs1(H(k, k-1)) <= smlnum {
					break
				}
				tst := abs1(H(k-1, k-1)) + abs1(H(k, k))
				if tst == 0 {
					if k-2 >= ilo {
						tst += math.Abs(real(H(k-1, k-2)))
					}
					if k+1 <= ihi {
						tst += math.Abs(real(H(k+1, k)))
					}
				}
				if math.Abs(real(H(k, k-1))) <= ulp*tst {
					ab := math.Max(abs1(H(k, k-1)), abs1(H(k-1, k)))
					ba := math.Min(abs1(H(k, k-1)), abs1(H(k-1, k)))
					aa := math.Max(abs1(H(k, k)), abs1(H(k-1, k-1)-H(k, k)))
					bb := math.Min(abs1(H(k, k)), abs1(H(k-1, k-1)-H(k, k)))
					s := aa + ab
					if ba*(ab/s) <= math.Max(smlnum, ulp*(bb*(aa/s))) {
						break
					}
				}
			}
			l = k
			if l > ilo {
				h[l*n+l-1] = 0
			}
			if l >= i {
				converged = true
				break
			}
			kdefl++
			if !wantt {
				i1, i2 = l, i
			}
			var t complex128
			switch {
			case kdefl%(2*kexsh) == 0:
				t = complex(dat1*math.Abs(real(H(i, i-1))), 0) + H(i, i)
			case kdefl%kexsh == 0:
				t = complex(dat1*math.Abs(real(H(l+1, l))), 0) + H(l, l)
			default:
				// Wilkinson's shift.
				t = H(i, i)
				u := cmplx.Sqrt(H(i-1, i)) * cmplx.Sqrt(H(i, i-1))
				if s := abs1(u); s != 0 {
					x := 0.5 * (H(i-1, i-1) - t)
					sx := abs1(x)
					s = math.Max(s, sx)
					cs := complex(s, 0)
					y := cs * cmplx.Sqrt((x/cs)*(x/cs)+(u/cs)*(u/cs))
					if sx > 0 {
						xs := x / complex(sx, 0)
						if real(xs)*real(y)+imag(xs)*imag(y) < 0 {
							y = -y
						}
					}
					t -= u * (u / (x + y))
				}
			}
			// Look for two consecutive small subdiagonal elements.
			m := i - 1
			for ; m > l; m-- {
				h11, h22 := H(m, m), H(m+1, m+1)
				h11s := h11 - t
				h21 := real(H(m+1, m))
				s := abs1(h11s) + math.Abs(h21)
				h11s /= complex(s, 0)
				h21 /= s
				v[0], v[1] = h11s, complex(h21, 0)
				h10 := real(H(m, m-1))
				if math.Abs(h10)*math.Abs(h21) <= ulp*(abs1(h11s)*(abs1(h11)+abs1(h22))) {
					break
				}
			}
			if m == l {
				h11s := H(l, l) - t
				h21 := real(H(l+1, l))
				s := abs1(h11s) + math.Abs(h21)
				v[0], v[1] = h11s/complex(s, 0), complex(h21/s, 0)
			}
			// The single-shift QR step.
			for k := m; k < i; k++ {
				if k > m {
					v[0], v[1] = H(k, k-1), H(k+1, k-1)
				}
				var t1 complex128
				v[0], t1 = larfg(2, v[0], v[1:], 1)
				if k > m {
					h[k*n+k-1], h[(k+1)*n+k-1] = v[0], 0
				}
				v2 := v[1]
				t2 := complex(real(t1*v2), 0)
				r0, r1 := h[k*n:k*n+n], h[(k+1)*n:(k+1)*n+n]
				for j := k; j <= i2; j++ {
					sum := cmplx.Conj(t1)*r0[j] + t2*r1[j]
					r0[j] -= sum
					r1[j] -= sum * v2
				}
				for j := i1; j <= min(k+2, i); j++ {
					row := h[j*n : j*n+n]
					sum := t1*row[k] + t2*row[k+1]
					row[k] -= sum
					row[k+1] -= sum * cmplx.Conj(v2)
				}
				if zt != nil {
					z0, z1 := zt[k*n:k*n+n], zt[(k+1)*n:(k+1)*n+n]
					for j := range z0 {
						sum := t1*z0[j] + t2*z1[j]
						z0[j] -= sum
						z1[j] -= sum * cmplx.Conj(v2)
					}
				}
				if k == m && m > l {
					// The step started below a small subdiagonal: rescale
					// so that h(m, m-1) stays real.
					temp := 1 - t1
					temp /= complex(cmplx.Abs(temp), 0)
					h[(m+1)*n+m] *= cmplx.Conj(temp)
					if m+2 <= i {
						h[(m+2)*n+m+1] *= temp
					}
					for j := m; j <= i; j++ {
						if j == m+1 {
							continue
						}
						if i2 > j {
							scaleRow(j, j+1, i2+1, temp)
						}
						scaleCol(j, i1, j, cmplx.Conj(temp))
						scaleZ(j, cmplx.Conj(temp))
					}
				}
			}
			// Make h(i, i-1) real.
			if temp := H(i, i-1); imag(temp) != 0 {
				rtemp := cmplx.Abs(temp)
				h[i*n+i-1] = complex(rtemp, 0)
				temp /= complex(rtemp, 0)
				if i2 > i {
					scaleRow(i, i+1, i2+1, cmplx.Conj(temp))
				}
				scaleCol(i, i1, i, temp)
				scaleZ(i, temp)
			}
		}
		if !converged {
			return false
		}
		w[i] = H(i, i)
		kdefl = 0
		i = l - 1
	}
	return true
}

// solve2 solves the 2×2 complex system [a b; c d]·y = r by Gaussian
// elimination with complete pivoting, perturbing pivots below smin to smin
// (as LAPACK dlaln2 does). The returned scale (<= 1) must also be applied
// to the rest of the vector: the solution is of scale·r, which keeps |y|
// below bignum.
func solve2(a, b, c, d, r1, r2 complex128, smin, bignum float64) (y1, y2 complex128, scale float64) {
	m := [4]complex128{a, b, c, d}
	p := 0
	for i := 1; i < 4; i++ {
		if abs1(m[i]) > abs1(m[p]) {
			p = i
		}
	}
	pr, pc := p/2, p%2 // pivot row and column
	or, oc := 1-pr, 1-pc
	rhs := [2]complex128{r1, r2}
	u11 := m[pr*2+pc]
	if abs1(u11) < smin {
		u11 = complex(smin, 0)
	}
	l21 := m[or*2+pc] / u11
	u12 := m[pr*2+oc]
	u22 := m[or*2+oc] - l21*u12
	if abs1(u22) < smin {
		u22 = complex(smin, 0)
	}
	b1 := rhs[pr]
	b2 := rhs[or] - l21*b1
	scale = 1
	if bn := math.Max(abs1(b1), abs1(b2)); bn > 1 && abs1(u22) < 1 && bn > bignum*abs1(u22) {
		scale = 1 / bn
		b1 *= complex(scale, 0)
		b2 *= complex(scale, 0)
	}
	var y [2]complex128
	y[oc] = b2 / u22
	y[pc] = (b1 - u12*y[oc]) / u11
	return y[0], y[1], scale
}

// eigvec solves (T(0:k,0:k) - lambda)·x = x(0:k) in place for the upper
// (quasi-)triangular n×n t, from row k-1 up: in a real t a nonzero
// subdiagonal entry marks a 2×2 block. x is rescaled as a whole when its
// entries would overflow (LAPACK trevc's scheme). cnorm holds the 1-norms
// of t's strictly upper columns.
func eigvec[T scalar](n int, t []T, k int, lambda complex128, x []complex128, cnorm []float64, smin, bignum float64) {
	tc := func(i, j int) complex128 {
		v := t[i*n+j]
		return complex(re(v), im(v))
	}
	rescale := func(s float64) {
		for i := range x {
			x[i] *= complex(s, 0)
		}
	}
	for j := k - 1; j >= 0; {
		if j > 0 && t[j*n+j-1] != 0 {
			y1, y2, s := solve2(tc(j-1, j-1)-lambda, tc(j-1, j), tc(j, j-1), tc(j, j)-lambda, x[j-1], x[j], smin, bignum)
			if s != 1 {
				rescale(s)
			}
			x[j-1], x[j] = y1, y2
			if xm := math.Max(abs1(y1), abs1(y2)); xm > 1 && math.Max(cnorm[j-1], cnorm[j]) > bignum/xm {
				rescale(1 / xm)
			}
			for r := 0; r < j-1; r++ {
				x[r] -= x[j-1]*tc(r, j-1) + x[j]*tc(r, j)
			}
			j -= 2
			continue
		}
		den := tc(j, j) - lambda
		if abs1(den) < smin {
			den = complex(smin, 0)
		}
		if xj := abs1(x[j]); xj > 1 && abs1(den) < 1 && xj > bignum*abs1(den) {
			rescale(1 / xj)
		}
		x[j] /= den
		if xm := abs1(x[j]); xm > 1 && cnorm[j] > bignum/xm {
			rescale(1 / xm)
		}
		for r := 0; r < j; r++ {
			x[r] -= x[j] * tc(r, j)
		}
		j--
	}
}

// eigReal is dgeev for one real n×n matrix a (destroyed): the eigenvalues
// and, when wantV, the unit eigenvectors as the columns of an n×n complex
// matrix, normalized as dgeev normalizes them (Euclidean norm 1, largest
// component real).
func eigReal(n int, a []float64, wantV bool) ([]complex128, []complex128, bool) {
	ilo, ihi, perm, scale := gebal(n, a)
	q := gehrd(n, ilo, ihi, a, wantV)
	wr, wi := make([]float64, n), make([]float64, n)
	for i := 0; i < n; i++ {
		if i < ilo || i > ihi {
			wr[i] = a[i*n+i]
		}
	}
	var zt []float64
	if wantV {
		zt = transposed(n, n, q)
	}
	if n > 0 && !lahqr(wantV, n, ilo, ihi, a, wr, wi, zt) {
		return nil, nil, false
	}
	w := make([]complex128, n)
	for i := range w {
		w[i] = complex(wr[i], wi[i])
	}
	if !wantV {
		return w, nil, true
	}
	// Eigenvectors of the Schur form T, as dtrevc stores them: a real one
	// in its column, a complex pair as its real and imaginary parts in two.
	cnorm := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := 0; i < j; i++ {
			cnorm[j] += math.Abs(a[i*n+j])
		}
	}
	smlnum := safmin * (float64(n) / ulp)
	bignum := (1 - ulp) / smlnum
	xs := make([]float64, n*n)
	x := make([]complex128, n)
	for ki := n - 1; ki >= 0; ki-- {
		clear(x)
		smin := math.Max(ulp*(math.Abs(wr[ki])+math.Abs(wi[ki])), smlnum)
		if wi[ki] == 0 {
			x[ki] = 1
			for k := 0; k < ki; k++ {
				x[k] = complex(-a[k*n+ki], 0)
			}
			eigvec(n, a, ki, complex(wr[ki], 0), x, cnorm, smin, bignum)
			normalizeInf(x[:ki+1])
			for k := 0; k <= ki; k++ {
				xs[k*n+ki] = real(x[k])
			}
			continue
		}
		// A complex pair at ki-1, ki: the vector of wr + i·|wi|.
		wpos := math.Abs(wi[ki])
		smin = math.Max(ulp*(math.Abs(wr[ki])+wpos), smlnum)
		if math.Abs(a[(ki-1)*n+ki]) >= math.Abs(a[ki*n+ki-1]) {
			x[ki-1] = 1
			x[ki] = complex(0, wpos/a[(ki-1)*n+ki])
		} else {
			x[ki-1] = complex(-wpos/a[ki*n+ki-1], 0)
			x[ki] = complex(0, 1)
		}
		for k := 0; k < ki-1; k++ {
			x[k] = -x[ki-1]*complex(a[k*n+ki-1], 0) - x[ki]*complex(a[k*n+ki], 0)
		}
		eigvec(n, a, ki-1, complex(wr[ki], wpos), x, cnorm, smin, bignum)
		normalizeInf(x[:ki+1])
		for k := 0; k <= ki; k++ {
			xs[k*n+ki-1], xs[k*n+ki] = real(x[k]), imag(x[k])
		}
		ki--
	}
	vr := make([]float64, n*n)
	gemm(n, n, n, 1, rm(zt, 0, n).t(), rm(xs, 0, n), 0, vr, n)
	gebak(n, ilo, ihi, perm, scale, vr)
	// Normalize as dgeev does, and assemble the complex vectors.
	vt := transposed(n, n, vr) // rows are the eigenvectors
	out := make([]complex128, n*n)
	for i := 0; i < n; i++ {
		ri := vt[i*n : i*n+n]
		if wi[i] == 0 {
			s := 1 / nrm2(n, ri, 1)
			for r, v := range ri {
				out[r*n+i] = complex(v*s, 0)
			}
			continue
		}
		rj := vt[(i+1)*n : (i+1)*n+n]
		s := 1 / lapy2(nrm2(n, ri, 1), nrm2(n, rj, 1))
		k, best := 0, -1.0
		for r := range ri {
			ri[r] *= s
			rj[r] *= s
			if v := ri[r]*ri[r] + rj[r]*rj[r]; v > best {
				k, best = r, v
			}
		}
		cs, sn, _ := lartg(ri[k], rj[k])
		rot(ri, rj, cs, sn)
		rj[k] = 0
		// wi[i] > 0: the pair is (re + i·im) for w[i], its conjugate for w[i+1].
		sg := math.Copysign(1, wi[i])
		for r := range ri {
			out[r*n+i] = complex(ri[r], sg*rj[r])
			out[r*n+i+1] = complex(ri[r], -sg*rj[r])
		}
		i++
	}
	return w, out, true
}

// normalizeInf scales x so its largest |re|+|im| is 1 (a zero x is left).
func normalizeInf(x []complex128) {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, abs1(v))
	}
	if m > 0 {
		for i := range x {
			x[i] /= complex(m, 0)
		}
	}
}

// eigCplx is zgeev for one complex n×n matrix a (destroyed).
func eigCplx(n int, a []complex128, wantV bool) ([]complex128, []complex128, bool) {
	ilo, ihi, perm, scale := gebal(n, a)
	q := gehrd(n, ilo, ihi, a, wantV)
	w := make([]complex128, n)
	for i := 0; i < n; i++ {
		if i < ilo || i > ihi {
			w[i] = a[i*n+i]
		}
	}
	var zt []complex128
	if wantV {
		zt = transposed(n, n, q)
	}
	if n > 0 && !zlahqr(wantV, n, ilo, ihi, a, w, zt) {
		return nil, nil, false
	}
	if !wantV {
		return w, nil, true
	}
	cnorm := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := 0; i < j; i++ {
			cnorm[j] += abs1(a[i*n+j])
		}
	}
	smlnum := safmin * (float64(n) / ulp)
	bignum := (1 - ulp) / smlnum
	xs := make([]complex128, n*n)
	x := make([]complex128, n)
	for ki := n - 1; ki >= 0; ki-- {
		clear(x)
		lam := a[ki*n+ki]
		smin := math.Max(ulp*abs1(lam), smlnum)
		x[ki] = 1
		for k := 0; k < ki; k++ {
			x[k] = -a[k*n+ki]
		}
		eigvec(n, a, ki, lam, x, cnorm, smin, bignum)
		normalizeInf(x[:ki+1])
		for k := 0; k <= ki; k++ {
			xs[k*n+ki] = x[k]
		}
	}
	vr := make([]complex128, n*n)
	gemm(n, n, n, 1, rm(zt, 0, n).t(), rm(xs, 0, n), 0, vr, n)
	gebak(n, ilo, ihi, perm, scale, vr)
	vt := transposed(n, n, vr)
	for i := 0; i < n; i++ {
		ri := vt[i*n : i*n+n]
		s := complex(1/nrm2(n, ri, 1), 0)
		k, best := 0, -1.0
		for r := range ri {
			ri[r] *= s
			if v := absSq(ri[r]); v > best {
				k, best = r, v
			}
		}
		ph := cmplx.Conj(ri[k]) / complex(math.Sqrt(best), 0)
		for r := range ri {
			ri[r] *= ph
		}
		ri[k] = complex(real(ri[k]), 0)
	}
	return w, transposed(n, n, vt), true
}

// Eig is numpy.linalg.eig: the eigenvalues w (..., M) and right
// eigenvectors v (..., M, M), as columns, of each square matrix of a. As in
// numpy 2, both are always complex (complex64 for single-precision inputs).
// Each vector has Euclidean norm 1 and its largest component real; for a
// real matrix, conjugate eigenvalues have conjugate vectors. The algorithm
// is LAPACK geev's: balancing, Hessenberg reduction, Francis double-shift QR
// (single-shift for complex input), back-substitution. An input with Inf or
// NaN gives an error wrapping ErrLinAlg, as in numpy.
func Eig(a *ndarray.Array) (w, v *ndarray.Array, err error) {
	return eigT(a, true)
}

// Eigvals is numpy.linalg.eigvals: the eigenvalues of Eig only.
func Eigvals(a *ndarray.Array) (*ndarray.Array, error) {
	w, _, err := eigT(a, false)
	return w, err
}

func eigT(a *ndarray.Array, wantV bool) (*ndarray.Array, *ndarray.Array, error) {
	o, err := square(a, "Eig")
	if err != nil {
		return nil, nil, err
	}
	n, cnt := o.n, o.count()
	var data []complex128
	var rdata []float64
	if o.cplx {
		data = load[complex128](o.a)
	} else {
		rdata = load[float64](o.a)
	}
	if !finite(data) || !finite(rdata) {
		return nil, nil, errNonFinite
	}
	ws := make([]complex128, cnt*n)
	var vs []complex128
	if wantV {
		vs = make([]complex128, cnt*n*n)
	}
	var fe firstErr
	parallelFor(cnt, 10*n*n*n, func(i int) {
		var w, v []complex128
		var ok bool
		if o.cplx {
			w, v, ok = eigCplx(n, data[i*n*n:(i+1)*n*n], wantV)
		} else {
			w, v, ok = eigReal(n, rdata[i*n*n:(i+1)*n*n], wantV)
		}
		if !ok {
			fe.set(i, ErrNoConvergence)
			return
		}
		copy(ws[i*n:], w)
		copy(vs[i*n*n:], v)
	})
	if fe.err != nil {
		return nil, nil, fe.err
	}
	w := wrap(ws, shapeOf(o.batch, n), o.single)
	if !wantV {
		return w, nil, nil
	}
	return w, wrap(vs, shapeOf(o.batch, n, n), o.single), nil
}

// finite reports whether every element of x is finite.
func finite[T scalar](x []T) bool {
	for _, v := range x {
		if r, i := re(v), im(v); math.IsInf(r, 0) || math.IsNaN(r) || math.IsInf(i, 0) || math.IsNaN(i) {
			return false
		}
	}
	return true
}
