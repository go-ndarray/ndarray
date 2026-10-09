package linalg

import (
	"math"
	"math/cmplx"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

// The algorithms are written once, generic over the two computation types.
// The helpers below switch on the type parameter; the compiler stencils
// float64 and complex128 separately and folds the switch away, so the real
// instantiation pays nothing for the complex case.
type scalar interface{ float64 | complex128 }

// Machine constants, as LAPACK's dlamch reports them for IEEE double.
const (
	eps    = 0x1p-53   // relative machine precision, dlamch('E')
	ulp    = 0x1p-52   // eps * base, dlamch('P')
	safmin = 0x1p-1022 // smallest normal number, dlamch('S')
)

func isCplx[T scalar]() bool {
	var z T
	_, ok := any(z).(complex128)
	return ok
}

func conj[T scalar](x T) T {
	if v, ok := any(x).(complex128); ok {
		return any(complex(real(v), -imag(v))).(T)
	}
	return x
}

func re[T scalar](x T) float64 {
	if v, ok := any(x).(complex128); ok {
		return real(v)
	}
	return any(x).(float64)
}

func im[T scalar](x T) float64 {
	if v, ok := any(x).(complex128); ok {
		return imag(v)
	}
	return 0
}

func fromReal[T scalar](r float64) T {
	var z T
	if _, ok := any(z).(complex128); ok {
		return any(complex(r, 0)).(T)
	}
	return any(r).(T)
}

// abs is the modulus |x|.
func abs[T scalar](x T) float64 {
	if v, ok := any(x).(complex128); ok {
		return cmplx.Abs(v)
	}
	return math.Abs(any(x).(float64))
}

// abs1 is |re x| + |im x|, LAPACK's cabs1 (|x| for a real x).
func abs1[T scalar](x T) float64 {
	if v, ok := any(x).(complex128); ok {
		return math.Abs(real(v)) + math.Abs(imag(v))
	}
	return math.Abs(any(x).(float64))
}

// rscale is x*r for a real r, without the cross terms (and NaNs from 0*Inf)
// of a complex product.
func rscale[T scalar](x T, r float64) T {
	if v, ok := any(x).(complex128); ok {
		return any(complex(real(v)*r, imag(v)*r)).(T)
	}
	return any(any(x).(float64) * r).(T)
}

// absSq is |x|².
func absSq[T scalar](x T) float64 {
	if v, ok := any(x).(complex128); ok {
		return real(v)*real(v) + imag(v)*imag(v)
	}
	f := any(x).(float64)
	return f * f
}

// nrm2 is the Euclidean norm of x[0], x[inc], ..., n elements, safe from
// overflow and underflow.
func nrm2[T scalar](n int, x []T, inc int) float64 {
	var ss float64
	for i := 0; i < n; i++ {
		ss += absSq(x[i*inc])
	}
	if ss > 0x1p-900 && ss < 0x1p900 {
		return math.Sqrt(ss)
	}
	if math.IsNaN(ss) {
		return ss
	}
	// Zero, underflowed or overflowed: rescale by the largest component.
	var big float64
	for i := 0; i < n; i++ {
		big = math.Max(big, math.Max(math.Abs(re(x[i*inc])), math.Abs(im(x[i*inc]))))
	}
	if big == 0 || math.IsInf(big, 1) {
		return big
	}
	ss = 0
	for i := 0; i < n; i++ {
		r, c := re(x[i*inc])/big, im(x[i*inc])/big
		ss += r*r + c*c
	}
	return big * math.Sqrt(ss)
}

// lapy2 is sqrt(x²+y²) without destructive overflow.
func lapy2(x, y float64) float64 { return math.Hypot(x, y) }

// lapy3 is sqrt(x²+y²+z²) without destructive overflow.
func lapy3(x, y, z float64) float64 {
	w := math.Max(math.Abs(x), math.Max(math.Abs(y), math.Abs(z)))
	if w == 0 {
		return math.Abs(x) + math.Abs(y) + math.Abs(z)
	}
	x, y, z = x/w, y/w, z/w
	return w * math.Sqrt(x*x+y*y+z*z)
}

// mview is a strided matrix operand: element (i, j) is d[off+i*rs+j*cs],
// conjugated when cj is set. Transposition swaps the strides.
type mview[T scalar] struct {
	d       []T
	off     int
	rs, cs  int
	conjVal bool
}

// rm views a row-major matrix with leading dimension ld starting at d[off].
func rm[T scalar](d []T, off, ld int) mview[T] { return mview[T]{d: d, off: off, rs: ld, cs: 1} }

func (v mview[T]) at(i, j int) T {
	x := v.d[v.off+i*v.rs+j*v.cs]
	if v.conjVal {
		return conj(x)
	}
	return x
}

// t is the transpose, h the conjugate transpose.
func (v mview[T]) t() mview[T] { v.rs, v.cs = v.cs, v.rs; return v }
func (v mview[T]) h() mview[T] {
	v.rs, v.cs = v.cs, v.rs
	v.conjVal = !v.conjVal
	return v
}

// gemmMin is the smallest m*n*k for which gemm hands the product to the
// packed SIMD GEMM of the ndarray kernels; below it a plain loop is faster
// than the packing.
var gemmMin = 1 << 13

// gemm computes C = beta*C + alpha*A*B for an m×k A, a k×n B and an m×n
// row-major C (leading dimension ldc, from c[0]).
func gemm[T scalar](m, n, k int, alpha T, a, b mview[T], beta T, c []T, ldc int) {
	if m == 0 || n == 0 {
		return
	}
	if beta != 1 {
		for i := 0; i < m; i++ {
			row := c[i*ldc : i*ldc+n]
			for j := range row {
				if beta == 0 {
					row[j] = 0
				} else {
					row[j] *= beta
				}
			}
		}
	}
	if k == 0 {
		return
	}
	if m*n*k >= gemmMin && min(m, n, k) >= 4 {
		fastGemm(m, n, k, alpha, a, b, c, ldc)
		return
	}
	for i := 0; i < m; i++ {
		ci := c[i*ldc : i*ldc+n]
		for p := 0; p < k; p++ {
			aip := alpha * a.at(i, p)
			if b.cs == 1 && !b.conjVal {
				bp := b.d[b.off+p*b.rs : b.off+p*b.rs+n]
				for j, x := range bp {
					ci[j] += aip * x
				}
				continue
			}
			for j := range ci {
				ci[j] += aip * b.at(p, j)
			}
		}
	}
}

// fastGemm adds alpha*A*B to C through the float64 packed GEMM: directly
// for real operands, as four real products for complex ones.
func fastGemm[T scalar](m, n, k int, alpha T, a, b mview[T], c []T, ldc int) {
	tmp := make([]float64, m*n)
	if cf, ok := any(c).([]float64); ok {
		av, bv := any(a).(mview[float64]), any(b).(mview[float64])
		kernels.MatMulStridedP(tmp, kmat(av), kmat(bv), m, k, n)
		al := any(alpha).(float64)
		for i := 0; i < m; i++ {
			row := cf[i*ldc : i*ldc+n]
			t := tmp[i*n : i*n+n]
			for j := range row {
				row[j] += al * t[j]
			}
		}
		return
	}
	ar, ai := split(m, k, a)
	br, bi := split(k, n, b)
	ti := make([]float64, m*n)
	ra, rb := kernels.Mat{Data: ar, RS: k, CS: 1}, kernels.Mat{Data: br, RS: n, CS: 1}
	ia, ib := kernels.Mat{Data: ai, RS: k, CS: 1}, kernels.Mat{Data: bi, RS: n, CS: 1}
	kernels.MatMulStridedP(tmp, ra, rb, m, k, n) // Re·Re
	kernels.MatMulStridedP(ti, ia, ib, m, k, n)  // Im·Im
	for i := range tmp {
		tmp[i] -= ti[i]
	}
	t3 := make([]float64, m*n)
	clear(ti)
	kernels.MatMulStridedP(ti, ra, ib, m, k, n) // Re·Im
	kernels.MatMulStridedP(t3, ia, rb, m, k, n) // Im·Re
	cz := any(c).([]complex128)
	al := any(alpha).(complex128)
	for i := 0; i < m; i++ {
		row := cz[i*ldc : i*ldc+n]
		for j := range row {
			row[j] += al * complex(tmp[i*n+j], ti[i*n+j]+t3[i*n+j])
		}
	}
}

func kmat(v mview[float64]) kernels.Mat {
	return kernels.Mat{Data: v.d, Off: v.off, RS: v.rs, CS: v.cs}
}

// split copies an r×c complex operand into contiguous real and imaginary
// parts.
func split[T scalar](r, c int, v mview[T]) (re, im []float64) {
	re, im = make([]float64, r*c), make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			z := any(v.at(i, j)).(complex128)
			re[i*c+j], im[i*c+j] = real(z), imag(z)
		}
	}
	return re, im
}

// trsmLeaf is the order at which the recursive triangular solves and
// factorizations switch to their unblocked loops.
var trsmLeaf = 32

// trsmLLU solves L·X = B in place of B, L n×n unit lower triangular (row-major
// at l, leading dimension ldl), B n×nrhs.
func trsmLLU[T scalar](n, nrhs int, l []T, ldl int, b []T, ldb int) {
	if n <= trsmLeaf {
		for i := 1; i < n; i++ {
			bi := b[i*ldb : i*ldb+nrhs]
			for k := 0; k < i; k++ {
				lik := l[i*ldl+k]
				bk := b[k*ldb : k*ldb+nrhs]
				for j, x := range bk {
					bi[j] -= lik * x
				}
			}
		}
		return
	}
	n1 := n / 2
	trsmLLU(n1, nrhs, l, ldl, b, ldb)
	gemm(n-n1, nrhs, n1, -1, rm(l, n1*ldl, ldl), rm(b, 0, ldb), 1, b[n1*ldb:], ldb)
	trsmLLU(n-n1, nrhs, l[n1*ldl+n1:], ldl, b[n1*ldb:], ldb)
}

// trsmLUN solves U·X = B in place of B, U n×n upper triangular with a
// nonzero diagonal, B n×nrhs.
func trsmLUN[T scalar](n, nrhs int, u []T, ldu int, b []T, ldb int) {
	if n <= trsmLeaf {
		for i := n - 1; i >= 0; i-- {
			bi := b[i*ldb : i*ldb+nrhs]
			for k := i + 1; k < n; k++ {
				uik := u[i*ldu+k]
				bk := b[k*ldb : k*ldb+nrhs]
				for j, x := range bk {
					bi[j] -= uik * x
				}
			}
			d := u[i*ldu+i]
			for j := range bi {
				bi[j] /= d
			}
		}
		return
	}
	n1 := n / 2
	trsmLUN(n-n1, nrhs, u[n1*ldu+n1:], ldu, b[n1*ldb:], ldb)
	gemm(n1, nrhs, n-n1, -1, rm(u, n1, ldu), rm(b, n1*ldb, ldb), 1, b, ldb)
	trsmLUN(n1, nrhs, u, ldu, b, ldb)
}

// trsmRLC solves X·Lᴴ = B in place of B, L n×n lower triangular with a
// nonzero diagonal, B m×n.
func trsmRLC[T scalar](m, n int, l []T, ldl int, b []T, ldb int) {
	if n <= trsmLeaf {
		for r := 0; r < m; r++ {
			x := b[r*ldb : r*ldb+n]
			for j := 0; j < n; j++ {
				s := x[j]
				lj := l[j*ldl : j*ldl+j]
				for k, v := range lj {
					s -= x[k] * conj(v)
				}
				x[j] = s / conj(l[j*ldl+j])
			}
		}
		return
	}
	n1 := n / 2
	trsmRLC(m, n1, l, ldl, b, ldb)
	gemm(m, n-n1, n1, -1, rm(b, 0, ldb), rm(l, n1*ldl, ldl).h(), 1, b[n1:], ldb)
	trsmRLC(m, n-n1, l[n1*ldl+n1:], ldl, b[n1:], ldb)
}

// herkLower computes the lower triangle of C -= A·Aᴴ, A n×k, C n×n; the
// strictly upper triangle may be overwritten with garbage.
func herkLower[T scalar](n, k int, a []T, lda int, c []T, ldc int) {
	if n <= 2*trsmLeaf {
		gemm(n, n, k, -1, rm(a, 0, lda), rm(a, 0, lda).h(), 1, c, ldc)
		return
	}
	n1 := n / 2
	herkLower(n1, k, a, lda, c, ldc)
	gemm(n-n1, n1, k, -1, rm(a, n1*lda, lda), rm(a, 0, lda).h(), 1, c[n1*ldc:], ldc)
	herkLower(n-n1, k, a[n1*lda:], lda, c[n1*ldc+n1:], ldc)
}

// rot applies the plane rotation [c s; -s c] to the rows x and y:
// x' = c x + s y, y' = c y - s x.
func rot[T scalar](x, y []T, c, s float64) {
	y = y[:len(x)]
	for i, xv := range x {
		yv := y[i]
		x[i] = rscale(xv, c) + rscale(yv, s)
		y[i] = rscale(yv, c) - rscale(xv, s)
	}
}

// lartg generates a plane rotation with c real: [c s; -s c]·[f; g] = [r; 0]
// (LAPACK dlartg, 3.10 and later).
func lartg(f, g float64) (c, s, r float64) {
	const rtmin = 0x1p-511 // sqrt(safmin)
	const rtmax = 0x1p+511 // sqrt(safmax/2)
	switch {
	case g == 0:
		return 1, 0, f
	case f == 0:
		return 0, math.Copysign(1, g), math.Abs(g)
	}
	f1, g1 := math.Abs(f), math.Abs(g)
	if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
		d := math.Sqrt(f*f + g*g)
		c = f1 / d
		r = math.Copysign(d, f)
		return c, g / r, r
	}
	u := math.Min(1/safmin, math.Max(safmin, math.Max(f1, g1)))
	fs, gs := f/u, g/u
	d := math.Sqrt(fs*fs + gs*gs)
	c = math.Abs(fs) / d
	r = math.Copysign(d, f)
	return c, gs / r, r * u
}

// larfg generates an elementary reflector H = I - tau·v·vᴴ with
// Hᴴ·[alpha; x] = [beta; 0], beta real, v = [1; x'] (LAPACK zlarfg): x (n-1
// elements, stride inc) is overwritten by x', and the new alpha (= beta) and
// tau are returned.
func larfg[T scalar](n int, alpha T, x []T, inc int) (T, T) {
	if n <= 0 {
		return alpha, 0
	}
	xnorm := nrm2(n-1, x, inc)
	alphr, alphi := re(alpha), im(alpha)
	if xnorm == 0 && alphi == 0 {
		return alpha, 0
	}
	beta := -math.Copysign(lapy3(alphr, alphi, xnorm), alphr)
	const sfm = safmin / eps
	knt := 0
	if math.Abs(beta) < sfm {
		// beta may be inaccurate: scale x and recompute.
		for {
			knt++
			for i := 0; i < n-1; i++ {
				x[i*inc] = rscale(x[i*inc], 1/sfm)
			}
			beta /= sfm
			alphi /= sfm
			alphr /= sfm
			if math.Abs(beta) >= sfm || knt >= 20 {
				break
			}
		}
		xnorm = nrm2(n-1, x, inc)
		beta = -math.Copysign(lapy3(alphr, alphi, xnorm), alphr)
	}
	var tau, scal T
	if isCplx[T]() {
		tau = any(complex((beta-alphr)/beta, -alphi/beta)).(T)
		scal = any(1 / (complex(alphr, alphi) - complex(beta, 0))).(T)
	} else {
		tau = fromReal[T]((beta - alphr) / beta)
		scal = fromReal[T](1 / (alphr - beta))
	}
	for i := 0; i < n-1; i++ {
		x[i*inc] *= scal
	}
	for ; knt > 0; knt-- {
		beta *= sfm
	}
	return fromReal[T](beta), tau
}

// larfL applies H = I - tau·v·vᴴ from the left to the m×n matrix C (v has
// m elements, contiguous): C -= tau·v·(vᴴC). w is scratch of length n.
func larfL[T scalar](m, n int, v []T, tau T, c []T, ldc int, w []T) {
	if tau == 0 || n == 0 {
		return
	}
	w = w[:n]
	clear(w)
	for i := 0; i < m; i++ {
		vi := conj(v[i])
		if vi == 0 {
			continue
		}
		row := c[i*ldc : i*ldc+n]
		for j, x := range row {
			w[j] += vi * x
		}
	}
	for i := 0; i < m; i++ {
		f := tau * v[i]
		if f == 0 {
			continue
		}
		row := c[i*ldc : i*ldc+n]
		for j := range row {
			row[j] -= f * w[j]
		}
	}
}

// larfR applies H = I - tau·v·vᴴ from the right to the m×n matrix C (v has
// n elements): C -= tau·(Cv)·vᴴ.
func larfR[T scalar](m, n int, v []T, tau T, c []T, ldc int) {
	if tau == 0 || m == 0 {
		return
	}
	v = v[:n]
	for i := 0; i < m; i++ {
		row := c[i*ldc : i*ldc+n]
		var s T
		for j, x := range row {
			s += x * v[j]
		}
		s *= tau
		if s == 0 {
			continue
		}
		for j := range row {
			row[j] -= s * conj(v[j])
		}
	}
}

// blockReflector is the compact WY form H1·H2···Hk = I - V·T·Vᴴ of k
// reflectors (LAPACK larft, forward, columnwise): V is m×k unit lower
// trapezoidal, T k×k upper triangular, both dense row-major.
type blockReflector[T scalar] struct {
	m, k int
	v, t []T
}

// newBlockReflector gathers the k reflectors stored below the diagonal of
// the m×k panel at a (leading dimension lda), with scalars tau.
func newBlockReflector[T scalar](m, k int, a []T, lda int, tau []T) blockReflector[T] {
	v := make([]T, m*k)
	for i := 0; i < m; i++ {
		for j := 0; j < k && j <= i; j++ {
			if i == j {
				v[i*k+j] = 1
			} else {
				v[i*k+j] = a[i*lda+j]
			}
		}
	}
	return blockReflectorFromV(m, k, v, tau)
}

func blockReflectorFromV[T scalar](m, k int, v []T, tau []T) blockReflector[T] {
	g := make([]T, k*k) // g = Vᴴ V
	gemm(k, k, m, 1, rm(v, 0, k).h(), rm(v, 0, k), 0, g, k)
	t := make([]T, k*k)
	for i := 0; i < k; i++ {
		ti := tau[i]
		// T(0:i, i) = -tau_i · T(0:i, 0:i) · (Vᴴ V)(0:i, i)
		for j := 0; j < i; j++ {
			var s T
			for p := j; p < i; p++ {
				s += t[j*k+p] * g[p*k+i]
			}
			t[j*k+i] = -ti * s
		}
		t[i*k+i] = ti
	}
	return blockReflector[T]{m: m, k: k, v: v, t: t}
}

// applyLeft overwrites the m×n matrix C with H·C (conjT false) or Hᴴ·C
// (conjT true), H = I - V·T·Vᴴ.
func (br blockReflector[T]) applyLeft(n int, c []T, ldc int, conjT bool) {
	m, k := br.m, br.k
	if n == 0 || k == 0 {
		return
	}
	w := make([]T, k*n) // W = Vᴴ C
	gemm(k, n, m, 1, rm(br.v, 0, k).h(), rm(c, 0, ldc), 0, w, n)
	tw := make([]T, k*n) // op(T) W
	tv := rm(br.t, 0, k)
	if conjT {
		tv = tv.h()
	}
	gemm(k, n, k, 1, tv, rm(w, 0, n), 0, tw, n)
	gemm(m, n, k, -1, rm(br.v, 0, k), rm(tw, 0, n), 1, c, ldc)
}

// applyRight overwrites the m×n matrix C with C·H (H = I - V·T·Vᴴ, V n×k).
func (br blockReflector[T]) applyRight(m int, c []T, ldc int) {
	n, k := br.m, br.k
	if m == 0 || k == 0 {
		return
	}
	w := make([]T, m*k) // W = C V
	gemm(m, k, n, 1, rm(c, 0, ldc), rm(br.v, 0, k), 0, w, k)
	wt := make([]T, m*k) // W T
	gemm(m, k, k, 1, rm(w, 0, k), rm(br.t, 0, k), 0, wt, k)
	gemm(m, n, k, -1, rm(wt, 0, k), rm(br.v, 0, k).h(), 1, c, ldc)
}
