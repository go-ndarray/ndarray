package linalg

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
)

// eps64 and eps32 are numpy's finfo(float64).eps and finfo(float32).eps.
const (
	eps64 = 0x1p-52
	eps32 = 0x1p-23
)

// LstsqResult is numpy.linalg.lstsq's result.
type LstsqResult struct {
	// X is the least-squares solution, (N,) for a vector b and (N, K) for a
	// matrix b; the minimum-norm one when a is rank deficient.
	X *ndarray.Array
	// Residuals holds the squared Euclidean norm of each column of b - a·X,
	// (K,) — (1,) for a vector b, as in numpy — when the rank is N and
	// M > N; otherwise it is empty, (0,).
	Residuals *ndarray.Array
	// Rank is the effective rank of a.
	Rank int
	// S holds a's singular values, decreasing, (min(M, N),).
	S *ndarray.Array
}

// Lstsq is numpy.linalg.lstsq(a, b) with numpy 2's default cutoff: the X
// minimizing ‖b - a·X‖ for an (M, N) a and a b of shape (M,) or (M, K).
// Singular values at most eps·max(M, N) times the largest are treated as
// zero (eps = 2⁻⁵²).
func Lstsq(a, b *ndarray.Array) (LstsqResult, error) {
	return lstsq(a, b, math.NaN())
}

// LstsqRcond is Lstsq with numpy's rcond: singular values at most rcond
// times the largest are treated as zero. As in LAPACK's gelsd, which numpy
// calls, an rcond outside (0, 1) means the machine precision 2⁻⁵³.
func LstsqRcond(a, b *ndarray.Array, rcond float64) (LstsqResult, error) {
	return lstsq(a, b, rcond)
}

func lstsq(a, b *ndarray.Array, rcond float64) (LstsqResult, error) {
	as, bs := a.Shape(), b.Shape()
	vec := len(bs) == 1
	if len(as) != 2 || (len(bs) != 2 && !vec) {
		return LstsqResult{}, fmt.Errorf("%w: Lstsq needs a 2-D a and a 1-D or 2-D b, got %d-D and %d-D", ErrShape, len(as), len(bs))
	}
	m, n := as[0], as[1]
	k := 1
	if !vec {
		k = bs[1]
	}
	if bs[0] != m {
		return LstsqResult{}, fmt.Errorf("%w: Lstsq a has %d rows, b %d", ErrShape, m, bs[0])
	}
	switch {
	case math.IsNaN(rcond):
		rcond = eps64 * float64(max(m, n))
	case rcond <= 0 || rcond >= 1:
		rcond = eps
	}
	cplx, single := precision(a, b)
	if cplx {
		return lstsqT[complex128](a, b, m, n, k, vec, rcond, single)
	}
	return lstsqT[float64](a, b, m, n, k, vec, rcond, single)
}

func lstsqT[T scalar](a, b *ndarray.Array, m, n, k int, vec bool, rcond float64, single bool) (LstsqResult, error) {
	ad, bd := load[T](a), load[T](b)
	p := min(m, n)
	s, u, vh, ok := svdOne(m, n, append([]T(nil), ad...), true, false)
	if !ok {
		return LstsqResult{}, fmt.Errorf("%w in linear least squares", ErrNoConvergence)
	}
	tol := 0.0
	if p > 0 {
		tol = rcond * s[0]
	}
	rank := 0
	for _, v := range s {
		if v > tol {
			rank++
		}
	}
	// X = V·S⁺·Uᴴ·b over the rank largest singular values.
	c := make([]T, p*k)
	gemm(p, k, m, 1, rm(u, 0, p).h(), rm(bd, 0, k), 0, c, k)
	for i := 0; i < p; i++ {
		row := c[i*k : i*k+k]
		for j := range row {
			if i < rank {
				row[j] = rscale(row[j], 1/s[i])
			} else {
				row[j] = 0
			}
		}
	}
	x := make([]T, n*k)
	gemm(n, k, p, 1, rm(vh, 0, n).h(), rm(c, 0, k), 0, x, k)
	var resid []float64
	if rank == n && m > n {
		// From the QR factorization: the residual is the part of Qᴴb below
		// row n, free of the cancellation in b - a·x.
		tau := make([]T, n)
		geqrf(m, n, ad, n, tau)
		v := make([]T, m)
		w := make([]T, k)
		for i := 0; i < n; i++ {
			v[0] = 1
			for r := i + 1; r < m; r++ {
				v[r-i] = ad[r*n+i]
			}
			larfL(m-i, k, v[:m-i], conj(tau[i]), bd[i*k:], k, w)
		}
		resid = make([]float64, k)
		for j := range resid {
			resid[j] = math.Pow(nrm2(m-n, bd[n*k+j:], k), 2)
		}
	}
	xs := []int{n, k}
	if vec {
		xs = xs[:1]
	}
	return LstsqResult{
		X:         wrap(x, xs, single),
		Residuals: wrap(resid, []int{len(resid)}, single),
		Rank:      rank,
		S:         wrap(s, []int{p}, single),
	}, nil
}

// Pinv is numpy.linalg.pinv(a) with numpy's default cutoff: the
// Moore–Penrose pseudo-inverse (..., N, M) of each matrix of a (..., M, N),
// from its SVD, singular values at most 1e-15 times the largest being
// treated as zero.
func Pinv(a *ndarray.Array) (*ndarray.Array, error) { return PinvRcond(a, 1e-15) }

// PinvRcond is Pinv with numpy's rcond: singular values at most rcond times
// the largest are treated as zero. numpy's rtol=None is
// PinvRcond(a, max(M, N)·eps).
func PinvRcond(a *ndarray.Array, rcond float64) (*ndarray.Array, error) {
	o, err := stacked(a, "Pinv")
	if err != nil {
		return nil, err
	}
	if o.m*o.n == 0 {
		// numpy returns an empty array of a's own dtype.
		return ndarray.ZerosOf(a.DType(), shapeOf(o.batch, o.n, o.m)...)
	}
	if o.cplx {
		return pinvT[complex128](o, rcond)
	}
	return pinvT[float64](o, rcond)
}

func pinvT[T scalar](o operand, rcond float64) (*ndarray.Array, error) {
	m, n, cnt := o.m, o.n, o.count()
	p := min(m, n)
	data := load[T](o.a)
	out := make([]T, cnt*n*m)
	var fe firstErr
	parallelFor(cnt, 8*m*n*p, func(i int) {
		s, u, vh, ok := svdOne(m, n, data[i*m*n:(i+1)*m*n], true, false)
		if !ok {
			fe.set(i, ErrNoConvergence)
			return
		}
		cut := rcond * s[0]
		// S⁺·Uᴴ, p×m.
		su := make([]T, p*m)
		for r := 0; r < p; r++ {
			if !(s[r] > cut) {
				continue
			}
			inv := 1 / s[r]
			for c := 0; c < m; c++ {
				su[r*m+c] = rscale(conj(u[c*p+r]), inv)
			}
		}
		gemm(n, m, p, 1, rm(vh, 0, n).h(), rm(su, 0, m), 0, out[i*n*m:], m)
	})
	if fe.err != nil {
		return nil, fe.err
	}
	return wrap(out, shapeOf(o.batch, n, m), o.single), nil
}

// singularValues returns the singular values of every matrix of o (k per
// matrix, decreasing), rounded to float32 when o is single precision.
func singularValues(o operand) ([]float64, error) {
	var s *ndarray.Array
	var err error
	if o.cplx {
		_, s, _, err = svdT[complex128](o, false, false)
	} else {
		_, s, _, err = svdT[float64](o, false, false)
	}
	if err != nil {
		return nil, err
	}
	d, _ := ndarray.Data[float64](s.AsType(ndarray.Float64))
	return d, nil
}

// MatrixRank is numpy.linalg.matrix_rank(A) with its default tolerance: the
// number of singular values of each matrix of a greater than
// S.max()·max(M, N)·eps, eps being the machine epsilon of the singular
// values' dtype (float32's for single-precision inputs). The result is
// Int64, shaped like a's batch dimensions. As in numpy, an array of fewer
// than two dimensions has rank 1 unless all its elements are zero.
func MatrixRank(a *ndarray.Array) (*ndarray.Array, error) { return matrixRank(a, math.NaN()) }

// MatrixRankTol is numpy.linalg.matrix_rank(A, tol): singular values
// greater than the absolute threshold tol count.
func MatrixRankTol(a *ndarray.Array, tol float64) (*ndarray.Array, error) {
	return matrixRank(a, tol)
}

func matrixRank(a *ndarray.Array, tol float64) (*ndarray.Array, error) {
	if len(a.Shape()) < 2 {
		nz := int64(0)
		if a.Any() {
			nz = 1
		}
		return ndarray.FromSlice([]int64{nz})
	}
	o, _ := stacked(a, "MatrixRank")
	s, err := singularValues(o)
	if err != nil {
		return nil, err
	}
	k := min(o.m, o.n)
	e := eps64
	if o.single {
		e = eps32
	}
	ranks := make([]int64, o.count())
	for i := range ranks {
		si := s[i*k : (i+1)*k]
		t := tol
		if math.IsNaN(t) {
			t = 0
			if k > 0 {
				t = si[0] * float64(max(o.m, o.n)) * e
			}
		}
		for _, v := range si {
			if v > t {
				ranks[i]++
			}
		}
	}
	return ndarray.FromSlice(ranks, o.batch...)
}

// Cond is numpy.linalg.cond: the condition number of each matrix of x in
// the norm p. The default P2 (or the zero Ord, numpy's None) and its
// negative are the ratio of the extreme singular values and take any
// (..., M, N); the other orders (Fro, Nuc, ±1, ±Inf) are norm(x)·norm(x⁻¹)
// and need square matrices. A singular matrix has condition number +Inf.
func Cond(x *ndarray.Array, p Ord) (*ndarray.Array, error) {
	o, err := stacked(x, "Cond")
	if err != nil {
		return nil, err
	}
	if o.m*o.n == 0 {
		return nil, fmt.Errorf("%w: Cond is not defined on empty matrices", ErrShape)
	}
	r := make([]float64, o.count())
	if p.kind == ordDefault || (p.kind == ordNum && math.Abs(p.p) == 2) {
		s, err := singularValues(o)
		if err != nil {
			return nil, err
		}
		k := min(o.m, o.n)
		for i := range r {
			if p.p == -2 {
				r[i] = s[i*k+k-1] / s[i*k]
			} else {
				r[i] = s[i*k] / s[i*k+k-1]
			}
		}
	} else {
		if o.m != o.n {
			return nil, fmt.Errorf("%w: Cond in this norm needs square matrices, got %d×%d", ErrShape, o.m, o.n)
		}
		if o.cplx {
			err = condT[complex128](o, p, r)
		} else {
			err = condT[float64](o, p, r)
		}
		if err != nil {
			return nil, err
		}
	}
	data := load[complex128](x)
	n2 := o.m * o.n
	for i, v := range r {
		// numpy turns a NaN from a singular matrix into +Inf, unless the
		// matrix itself holds a NaN.
		if math.IsNaN(v) && finiteOrInf(data[i*n2:(i+1)*n2]) {
			r[i] = math.Inf(1)
		}
	}
	return wrap(r, o.batch, o.single), nil
}

// finiteOrInf reports whether x holds no NaN.
func finiteOrInf(x []complex128) bool {
	for _, v := range x {
		if math.IsNaN(real(v)) || math.IsNaN(imag(v)) {
			return false
		}
	}
	return true
}

func condT[T scalar](o operand, p Ord, r []float64) error {
	n := o.n
	d := load[T](o.a)
	for i := range r {
		x := d[i*n*n : (i+1)*n*n]
		nx, err := matNorm(n, n, x, p)
		if err != nil {
			return err
		}
		lu := append([]T(nil), x...)
		piv := make([]int, n)
		if getrf(n, n, lu, n, piv) >= 0 {
			r[i] = math.NaN()
			continue
		}
		inv := make([]T, n*n)
		for j := 0; j < n; j++ {
			inv[j*n+j] = 1
		}
		getrs(n, n, lu, piv, inv)
		ni, _ := matNorm(n, n, inv, p)
		r[i] = nx * ni
	}
	return nil
}
