package linalg

import (
	"math"
	"sort"
)

// The divide-and-conquer SVD of a bidiagonal matrix (Gu and Eisenstat's
// method, as LAPACK's dbdsdc/dlasd0-4, which numpy's gesdd calls). A
// bidiagonal B is split at a middle row into two smaller bidiagonals; from
// their SVDs, B's is that of an "arrow" matrix — a first row z over a
// diagonal — whose singular values are the roots of a secular equation and
// whose singular vectors are explicit. Products with the halves' vectors
// are GEMMs, which is where the work is; the QR iteration it replaces
// spends O(n³) in plane rotations instead.

// bdsdc returns the SVD B = U·diag(d)·VT of the n×n upper bidiagonal (d, e):
// singular values decreasing into d, U and VT n×n row-major. e is
// destroyed.
func bdsdc(n int, d, e []float64) (u, vt []float64, ok bool) {
	return bdRec(n, 0, d, e)
}

// bdRec is the SVD of the n×(n+sqre) upper bidiagonal (d, e) — e has
// n-1+sqre entries — as U (n×n) and VT ((n+sqre)×(n+sqre)), the singular
// values decreasing into d. With sqre = 1, the last row of VT spans the
// null space.
func bdRec(n, sqre int, d, e []float64) ([]float64, []float64, bool) {
	if n <= dcLeaf {
		return bdLeaf(n, sqre, d, e)
	}
	k := n / 2
	nl, nr := k, n-k-1
	alpha, beta := d[k], e[k]
	u1, vt1, ok1 := bdRec(nl, 1, d[:nl], e[:nl])
	u2, vt2, ok2 := bdRec(nr, sqre, d[k+1:], e[k+1:])
	if !ok1 || !ok2 {
		return nil, nil, false
	}
	u, vt := bdMerge(n, sqre, k, d, alpha, beta, u1, vt1, u2, vt2)
	return u, vt, true
}

// bdLeaf solves a small bidiagonal by implicit QR (bdsqr). A non-square
// one is first made square: plane rotations on the columns chase its last
// entry up and out (LAPACK dlasdq), and only VT sees them.
func bdLeaf(n, sqre int, d, e []float64) ([]float64, []float64, bool) {
	m := n + sqre
	w := make([]float64, m*m)
	for i := 0; i < m; i++ {
		w[i*m+i] = 1
	}
	if sqre == 1 && n > 0 {
		f := e[n-1] // the entry to remove, at (i, n)
		for i := n - 1; i >= 0; i-- {
			c, s, r := lartg(d[i], f)
			d[i] = r
			rot(w[i*m:(i+1)*m], w[n*m:(n+1)*m], c, s)
			if i > 0 {
				f = -s * e[i-1]
				e[i-1] *= c
			}
		}
	}
	ut := make([]float64, n*n)
	for i := 0; i < n; i++ {
		ut[i*n+i] = 1
	}
	if !bdsqr(n, d, e, w, m, ut, n) {
		return nil, nil, false
	}
	return transposed(n, n, ut), w, true
}

// bdMerge forms the SVD of the n×(n+sqre) bidiagonal split at row k (dlasd1):
// d[:k] and d[k+1:] hold the halves' singular values, (u1, vt1) and
// (u2, vt2) their vectors, alpha and beta row k's two entries.
func bdMerge(n, sqre, k int, d []float64, alpha, beta float64, u1, vt1, u2, vt2 []float64) ([]float64, []float64) {
	m := n + sqre
	nl, nr := k, n-k-1
	m1, m2 := nl+1, nr+sqre
	// Scale to avoid overflow and underflow.
	orgnrm := math.Max(math.Abs(alpha), math.Abs(beta))
	for i := 0; i < n; i++ {
		if i != k {
			orgnrm = math.Max(orgnrm, math.Abs(d[i]))
		}
	}
	if orgnrm == 0 {
		orgnrm = 1
	}
	alpha /= orgnrm
	beta /= orgnrm
	// The bases, transposed (one vector per row), in the arrow's order: row
	// 0 is the arrow's first row (left: e_k; right: the halves' null
	// vectors), then the halves' singular pairs.
	ut := make([]float64, n*n)
	vtf := make([]float64, m*m)
	pole := make([]float64, n)
	z := make([]float64, n)
	ut[k] = 1
	for i := 0; i < nl; i++ {
		row := ut[(1+i)*n:]
		for r := 0; r < nl; r++ {
			row[r] = u1[r*nl+i]
		}
		copy(vtf[(1+i)*m:(1+i)*m+m1], vt1[i*m1:(i+1)*m1])
		pole[1+i] = d[i] / orgnrm
		z[1+i] = alpha * vt1[i*m1+nl]
	}
	for i := 0; i < nr; i++ {
		row := ut[(1+nl+i)*n:]
		for r := 0; r < nr; r++ {
			row[k+1+r] = u2[r*nr+i]
		}
		copy(vtf[(1+nl+i)*m+m1:(2+nl+i)*m], vt2[i*m2:(i+1)*m2])
		pole[1+nl+i] = d[k+1+i] / orgnrm
		z[1+nl+i] = beta * vt2[i*m2]
	}
	null1 := vt1[nl*m1 : (nl+1)*m1]
	a1 := alpha * null1[nl]
	if sqre == 1 {
		// Two null vectors: rotate them into one that meets row k and one
		// that stays null.
		null2 := vt2[nr*m2 : (nr+1)*m2]
		a2 := beta * null2[0]
		r0 := lapy2(a1, a2)
		c, s := 1.0, 0.0
		if r0 != 0 {
			c, s = a1/r0, a2/r0
		}
		for i, v := range null1 {
			vtf[i] = c * v
			vtf[(m-1)*m+i] = -s * v
		}
		for i, v := range null2 {
			vtf[m1+i] = s * v
			vtf[(m-1)*m+m1+i] = c * v
		}
		z[0] = r0
	} else {
		copy(vtf[:m1], null1)
		z[0] = a1
	}
	// Sort the poles ascending; the arrow's 0 stays first.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx[1:], func(a, b int) bool { return pole[idx[1+a]] < pole[idx[1+b]] })
	// Deflation (dlasd2).
	tol := 8 * eps * math.Max(math.Max(math.Abs(alpha), math.Abs(beta)), pole[idx[n-1]])
	urow := func(g int) []float64 { return ut[g*n : (g+1)*n] }
	vrow := func(g int) []float64 { return vtf[g*m : (g+1)*m] }
	keep := []int{0}
	var defl []int
	pj := -1
	for _, g := range idx[1:] {
		if math.Abs(z[g]) <= tol {
			defl = append(defl, g)
			continue
		}
		if pj < 0 {
			pj = g
			continue
		}
		if math.Abs(pole[g]-pole[pj]) <= tol {
			// Close poles: a rotation zeroes z at pj.
			s, c := z[pj], z[g]
			tau := lapy2(c, s)
			c /= tau
			s = -s / tau
			z[g], z[pj] = tau, 0
			rot(urow(pj), urow(g), c, s)
			rot(vrow(pj), vrow(g), c, s)
			defl = append(defl, pj)
		} else {
			keep = append(keep, pj)
		}
		pj = g
	}
	if pj >= 0 {
		keep = append(keep, pj)
	}
	if math.Abs(z[0]) <= tol {
		z[0] = tol
	}
	kk := len(keep)
	sk, zk := make([]float64, kk), make([]float64, kk)
	for i, g := range keep {
		sk[i], zk[i] = pole[g], z[g]
	}
	if kk > 1 && sk[1] <= tol/2 {
		sk[1] = tol / 2
	}
	// The secular equation in the squared poles.
	p := make([]float64, kk)
	for i, v := range sk {
		p[i] = v * v
	}
	diff := func(i, o int) float64 { return (sk[i] - sk[o]) * (sk[i] + sk[o]) }
	delta := make([]float64, kk*kk)
	sigma := make([]float64, kk)
	for j := 0; j < kk; j++ {
		sigma[j] = math.Sqrt(secular(p, zk, 1, j, diff, delta[j*kk:(j+1)*kk]))
	}
	zh := zHat(p, zk, delta, diff)
	// The arrow's singular vectors: right ∝ zhat_i/(s_i² - σ²), left ∝
	// (-1, s_i·zhat_i/(s_i² - σ²)).
	vr := make([]float64, kk*kk)
	ul := make([]float64, kk*kk)
	for j := 0; j < kk; j++ {
		r, l := vr[j*kk:(j+1)*kk], ul[j*kk:(j+1)*kk]
		for i := range r {
			r[i] = zh[i] / delta[j*kk+i]
			l[i] = sk[i] * r[i]
		}
		l[0] = -1
		sr, sl := 1/nrm2(kk, r, 1), 1/nrm2(kk, l, 1)
		for i := range r {
			r[i] *= sr
			l[i] *= sl
		}
	}
	gu, gv := make([]float64, kk*n), make([]float64, kk*m)
	for i, g := range keep {
		copy(gu[i*n:(i+1)*n], urow(g))
		copy(gv[i*m:(i+1)*m], vrow(g))
	}
	outU := make([]float64, n*n)
	outV := make([]float64, m*m)
	gemm(kk, n, kk, 1, rm(ul, 0, kk), rm(gu, 0, n), 0, outU, n)
	gemm(kk, m, kk, 1, rm(vr, 0, kk), rm(gv, 0, m), 0, outV, m)
	vals := append(sigma, make([]float64, len(defl))...)
	for i, g := range defl {
		vals[kk+i] = pole[g]
		copy(outU[(kk+i)*n:(kk+i+1)*n], urow(g))
		copy(outV[(kk+i)*m:(kk+i+1)*m], vrow(g))
	}
	if sqre == 1 {
		copy(outV[(m-1)*m:], vtf[(m-1)*m:])
	}
	// Sort decreasing and unscale.
	ord := make([]int, n)
	for i := range ord {
		ord[i] = i
	}
	sort.SliceStable(ord, func(a, b int) bool { return vals[ord[a]] > vals[ord[b]] })
	u := make([]float64, n*n)
	vt := make([]float64, m*m)
	for i, o := range ord {
		d[i] = vals[o] * orgnrm
		for r := 0; r < n; r++ {
			u[r*n+i] = outU[o*n+r]
		}
		copy(vt[i*m:(i+1)*m], outV[o*m:(o+1)*m])
	}
	if sqre == 1 {
		copy(vt[(m-1)*m:], outV[(m-1)*m:])
	}
	return u, vt
}
