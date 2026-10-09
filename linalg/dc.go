package linalg

import (
	"math"
	"sort"
)

// The divide-and-conquer eigensolver for symmetric tridiagonal matrices
// (Cuppen's method in the form of LAPACK dstedc/dlaed0-4, with Gu and
// Eisenstat's recomputation of the updating vector, which keeps the
// eigenvectors numerically orthogonal). Its work is dominated by matrix
// products, which run on the packed GEMM; the implicit QL/QR iteration it
// replaces spends O(n³) in plane rotations instead.

// dcLeaf is the order below which subproblems are solved by steqr.
var dcLeaf = 25

// stedc returns the eigenvalues of the symmetric tridiagonal (d, e),
// ascending (overwriting d), and the transpose of its eigenvector matrix:
// row k of the n×n result is the eigenvector of d[k]. e is destroyed.
func stedc(n int, d, e []float64) ([]float64, bool) {
	if n <= dcLeaf {
		zt := make([]float64, n*n)
		for i := 0; i < n; i++ {
			zt[i*n+i] = 1
		}
		return zt, steqr(n, d, e, zt, n)
	}
	m := n / 2
	rho := e[m-1]
	d[m-1] -= math.Abs(rho)
	d[m] -= math.Abs(rho)
	qt1, ok1 := stedc(m, d[:m], e[:m-1])
	qt2, ok2 := stedc(n-m, d[m:], e[m:])
	if !ok1 || !ok2 {
		return nil, false
	}
	return dcMerge(n, m, d, rho, qt1, qt2), true
}

// dcMerge solves T = diag(Q1·D1·Q1ᵀ, Q2·D2·Q2ᵀ) + |rho|·v·vᵀ (dlaed1): the
// eigenvalues D of the halves are in d[:m] and d[m:], ascending, and the
// transposed eigenvector matrices in qt1, qt2.
func dcMerge(n, m int, d []float64, rho float64, qt1, qt2 []float64) []float64 {
	n2 := n - m
	// Q, transposed, and the updating vector z = Qᵀv, of norm one.
	qt := make([]float64, n*n)
	z := make([]float64, n)
	sgn := math.Copysign(1, rho)
	for g := 0; g < m; g++ {
		copy(qt[g*n:g*n+m], qt1[g*m:(g+1)*m])
		z[g] = qt1[g*m+m-1] / math.Sqrt2
	}
	for g := 0; g < n2; g++ {
		copy(qt[(m+g)*n+m:(m+g+1)*n], qt2[g*n2:(g+1)*n2])
		z[m+g] = sgn * qt2[g*n2] / math.Sqrt2
	}
	rho = 2 * math.Abs(rho)
	// Sort the poles (each half is ascending already).
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return d[idx[a]] < d[idx[b]] })
	ds, zs := make([]float64, n), make([]float64, n)
	for i, g := range idx {
		ds[i], zs[i] = d[g], z[g]
	}
	row := func(i int) []float64 { g := idx[i]; return qt[g*n : g*n+n] }
	// Deflation (dlaed2): a negligible component of z, or two poles close
	// enough that a rotation zeroes one component, leave an eigenpair of the
	// halves unchanged.
	zmax, dmax := 0.0, 0.0
	for i := range ds {
		zmax = math.Max(zmax, math.Abs(zs[i]))
		dmax = math.Max(dmax, math.Abs(ds[i]))
	}
	tol := 8 * eps * math.Max(dmax, zmax)
	var keep, defl []int // positions in the sorted order
	pj := -1
	for j := 0; j < n; j++ {
		if rho*math.Abs(zs[j]) <= tol {
			defl = append(defl, j)
			continue
		}
		if pj < 0 {
			pj = j
			continue
		}
		s, c := zs[pj], zs[j]
		tau := lapy2(c, s)
		t := ds[j] - ds[pj]
		c /= tau
		s = -s / tau
		if math.Abs(t*c*s) <= tol {
			zs[j], zs[pj] = tau, 0
			rot(row(pj), row(j), c, s)
			t = ds[pj]*c*c + ds[j]*s*s
			ds[j] = ds[pj]*s*s + ds[j]*c*c
			ds[pj] = t
			defl = append(defl, pj)
		} else {
			keep = append(keep, pj)
		}
		pj = j
	}
	if pj >= 0 {
		keep = append(keep, pj)
	}
	k := len(keep)
	dk, zk := make([]float64, k), make([]float64, k)
	for i, p := range keep {
		dk[i], zk[i] = ds[p], zs[p]
	}
	// The secular equation, one root at a time; delta[j*k+i] = dk[i] - lambda_j.
	lam := make([]float64, k)
	delta := make([]float64, k*k)
	diff := func(i, o int) float64 { return dk[i] - dk[o] }
	for j := 0; j < k; j++ {
		lam[j] = secular(dk, zk, rho, j, diff, delta[j*k:(j+1)*k])
	}
	// Gu–Eisenstat: the z for which the computed roots are exact, then the
	// eigenvectors (zhat_i / (d_i - lambda_j)), normalized.
	zhat := zHat(dk, zk, delta, diff)
	u := make([]float64, k*k) // row j: the coefficients of eigenvector j
	for j := 0; j < k; j++ {
		uj := u[j*k : (j+1)*k]
		for i := range uj {
			uj[i] = zhat[i] / delta[j*k+i]
		}
		s := 1 / nrm2(k, uj, 1)
		for i := range uj {
			uj[i] *= s
		}
	}
	// New eigenvectors: rows of U·Q[keep], then the deflated ones.
	gath := make([]float64, k*n)
	for i, p := range keep {
		copy(gath[i*n:(i+1)*n], row(p))
	}
	out := make([]float64, n*n)
	gemm(k, n, k, 1, rm(u, 0, k), rm(gath, 0, n), 0, out, n)
	vals := append(lam, make([]float64, len(defl))...)
	for i, p := range defl {
		vals[k+i] = ds[p]
		copy(out[(k+i)*n:(k+i+1)*n], row(p))
	}
	// Sort ascending.
	ord := make([]int, n)
	for i := range ord {
		ord[i] = i
	}
	sort.SliceStable(ord, func(a, b int) bool { return vals[ord[a]] < vals[ord[b]] })
	res := make([]float64, n*n)
	for i, o := range ord {
		d[i] = vals[o]
		copy(res[i*n:(i+1)*n], out[o*n:(o+1)*n])
	}
	return res
}

// secular returns the j-th smallest root lambda of the secular equation
// 1/rho + Σ z_i²/(p_i - lambda) = 0 (poles p ascending and distinct, z
// without zeros, rho > 0), and stores p_i - lambda into delta, each
// computed from the pole nearest the root so that it is accurate to a few
// ulps even when the root is very close to a pole (LAPACK dlaed4's and
// dlasd4's requirement). diff(i, o) must return p_i - p_o accurately: for
// the squared poles of the SVD, (d_i - d_o)(d_i + d_o), not a difference of
// rounded squares.
//
// The iteration is the rational two-pole model of Bunch, Nielsen and
// Sorensen — the sum over the poles left of the root and the sum over those
// right of it are each replaced by a constant plus one pole, matching value
// and slope — safeguarded by a bracket that bisection falls back on.
func secular(p, z []float64, rho float64, j int, diff func(i, o int) float64, delta []float64) float64 {
	k := len(p)
	dd := delta // the poles relative to the origin
	var origin int
	var lo, hi float64
	if j == k-1 {
		origin = k - 1
		for _, v := range z {
			hi += v * v
		}
		hi *= rho
	} else {
		gap := diff(j+1, j)
		f := 1 / rho
		for i := range p {
			f += z[i] * z[i] / (diff(i, j) - gap/2)
		}
		if f >= 0 {
			origin, lo, hi = j, 0, gap/2
		} else {
			origin, lo, hi = j+1, -gap/2, 0
		}
	}
	for i := range p {
		dd[i] = diff(i, origin)
	}
	var p1, p2 float64 // the poles bounding the root
	if j == k-1 {
		p1 = dd[j]
	} else {
		p1, p2 = dd[j], dd[j+1]
	}
	tau := (lo + hi) / 2
	for it := 0; it < 200; it++ {
		var psi, dpsi, phi, dphi float64
		for i := 0; i <= j; i++ {
			t := z[i] / (dd[i] - tau)
			psi += z[i] * t
			dpsi += t * t
		}
		for i := j + 1; i < k; i++ {
			t := z[i] / (dd[i] - tau)
			phi += z[i] * t
			dphi += t * t
		}
		f := 1/rho + psi + phi
		if f < 0 {
			lo = tau
		} else {
			hi = tau
		}
		erretm := 8*(math.Abs(psi)+math.Abs(phi)+1/rho) + math.Abs(tau)*(dpsi+dphi)
		if math.Abs(f) <= eps*erretm || hi-lo <= 2*eps*math.Max(math.Abs(lo), math.Abs(hi)) {
			break
		}
		// The two-pole model: c + b1/(p1-x) + b2/(p2-x) = 0.
		b1 := dpsi * (p1 - tau) * (p1 - tau)
		c := 1/rho + psi - dpsi*(p1-tau)
		var x float64
		if j == k-1 {
			c += phi
			x = p1 + b1/c
		} else {
			b2 := dphi * (p2 - tau) * (p2 - tau)
			c += phi - dphi*(p2-tau)
			x = quadRoot(c, b1, b2, p1, p2, lo, hi)
		}
		if !(x > lo && x < hi) {
			x = (lo + hi) / 2
		}
		tau = x
	}
	for i := range dd {
		dd[i] -= tau
	}
	return p[origin] + tau
}

// zHat returns Gu and Eisenstat's updating vector: the z for which the
// computed roots are the exact eigenvalues of diag(p) + rho·z·zᵀ, with the
// signs of z. delta[j*k+i] is p_i - lambda_j.
func zHat(p, z, delta []float64, diff func(i, o int) float64) []float64 {
	k := len(p)
	zh := make([]float64, k)
	for i := 0; i < k; i++ {
		w := delta[i*k+i]
		for j := 0; j < k; j++ {
			if j != i {
				w *= delta[j*k+i] / diff(i, j)
			}
		}
		zh[i] = math.Copysign(math.Sqrt(math.Abs(w)), z[i])
	}
	return zh
}

// quadRoot solves c·(p1-x)(p2-x) + b1·(p2-x) + b2·(p1-x) = 0 for the root
// between lo and hi, or returns NaN.
func quadRoot(c, b1, b2, p1, p2, lo, hi float64) float64 {
	a := c
	b := -(c*(p1+p2) + b1 + b2)
	cc := c*p1*p2 + b1*p2 + b2*p1
	if a == 0 {
		return -cc / b
	}
	disc := b*b - 4*a*cc
	if disc < 0 {
		return math.NaN()
	}
	q := -0.5 * (b + math.Copysign(math.Sqrt(disc), b))
	r1, r2 := q/a, cc/q
	if r1 > lo && r1 < hi {
		return r1
	}
	return r2
}
