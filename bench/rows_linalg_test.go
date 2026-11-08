package bench

import (
	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/linalg"
)

// Linear algebra on 64 x 64 matrices; the reference is numpy.linalg on
// OpenBLAS/LAPACK with one thread.
const lm = 64

var (
	lGen = dgen(lm)
	lDom = ddom(lm)
	lSPD = dspd(lm)
	lRHS = dmat(0, 1, lm, lm/8, 13)
)

// lin is one row: expr is the numpy.linalg call over the parameters params.
func lin(name, params, expr string, args []ds, body func(a []*ndarray.Array)) row {
	return row{name: "linalg." + name, fn: "linalg." + name, ref: "numpy np.linalg." + expr, args: args,
		call: "lambda " + params + ": np.linalg." + expr,
		body: func(a []*ndarray.Array) func() { return func() { body(a) } }}
}

func lin1(name, expr string, in ds, h func(a *ndarray.Array) (*ndarray.Array, error)) row {
	return lin(name, "a", expr, []ds{in}, func(a []*ndarray.Array) { sinkA = must(h(a[0])) })
}

func linalgRows() []row {
	two := func(x ...*ndarray.Array) { sinkAs = x }
	return []row{
		lin1("Det", "det(a)", lDom, linalg.Det),
		lin1("Inv", "inv(a)", lDom, linalg.Inv),
		lin1("Eigvals", "eigvals(a)", lGen, linalg.Eigvals),
		lin1("Pinv", "pinv(a)", lDom, linalg.Pinv),
		lin1("SVDVals", "svd(a, compute_uv=False)", lDom, linalg.SVDVals),
		lin1("MatrixRank", "matrix_rank(a)", lDom, linalg.MatrixRank),
		lin1("Cholesky", "cholesky(a)", lSPD, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.Cholesky(a, linalg.Lower) }),
		lin1("Eigvalsh", "eigvalsh(a)", lSPD, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.Eigvalsh(a, linalg.Lower) }),
		lin1("Cond", "cond(a)", lDom, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.Cond(a, linalg.Ord{}) }),
		lin1("Norm", "norm(a)", lDom, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.Norm(a, linalg.Ord{}, nil, false) }),
		lin1("MatrixRankTol", "matrix_rank(a, tol=1e-8)", lDom, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.MatrixRankTol(a, 1e-8) }),
		lin1("PinvRcond", "pinv(a, rcond=1e-10)", lDom, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.PinvRcond(a, 1e-10) }),
		lin1("MatrixPower", "matrix_power(a, 5)", lDom, func(a *ndarray.Array) (*ndarray.Array, error) { return linalg.MatrixPower(a, 5) }),
		lin("Eig", "a", "eig(a)", []ds{lGen}, func(a []*ndarray.Array) { w, v, err := linalg.Eig(a[0]); chk(err); two(w, v) }),
		lin("Eigh", "a", "eigh(a)", []ds{lSPD}, func(a []*ndarray.Array) { w, v, err := linalg.Eigh(a[0], linalg.Lower); chk(err); two(w, v) }),
		lin("QR", "a", "qr(a)", []ds{lDom}, func(a []*ndarray.Array) { q, r, err := linalg.QR(a[0], linalg.ModeReduced); chk(err); two(q, r) }),
		lin("SVD", "a", "svd(a, full_matrices=False)", []ds{lDom}, func(a []*ndarray.Array) { u, s, vh, err := linalg.SVD(a[0], false); chk(err); two(u, s, vh) }),
		lin("Slogdet", "a", "slogdet(a)", []ds{lDom}, func(a []*ndarray.Array) { s, l, err := linalg.Slogdet(a[0]); chk(err); two(s, l) }),
		lin("Solve", "a, b", "solve(a, b)", []ds{lDom, lRHS}, func(a []*ndarray.Array) { sinkA = must(linalg.Solve(a[0], a[1])) }),
		lin("Lstsq", "a, b", "lstsq(a, b, rcond=None)", []ds{lDom, lRHS}, func(a []*ndarray.Array) { r, err := linalg.Lstsq(a[0], a[1]); chk(err); sinkA = r.X }),
		lin("LstsqRcond", "a, b", "lstsq(a, b, rcond=1e-10)", []ds{lDom, lRHS}, func(a []*ndarray.Array) { r, err := linalg.LstsqRcond(a[0], a[1], 1e-10); chk(err); sinkA = r.X }),
		lin("MultiDot", "a, b, c", "multi_dot([a, b, c])", []ds{lDom, lSPD, lGen}, func(a []*ndarray.Array) { sinkA = must(linalg.MultiDot(a...)) }),
	}
}
