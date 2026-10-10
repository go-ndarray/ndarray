package linalg

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// These tests check the identities that define each result on inputs the
// fixture keeps small to stay small: orders above the blocking and
// divide-and-conquer thresholds, stacks large enough to run in parallel,
// and matrices built to reach the rarely taken branches (deflation,
// rescaling, exceptional shifts, failure to converge).

func randC(r *rand.Rand, cplx bool, shape ...int) *ndarray.Array {
	n := 1
	for _, s := range shape {
		n *= s
	}
	d := make([]complex128, n)
	for i := range d {
		d[i] = complex(r.NormFloat64(), 0)
		if cplx {
			d[i] += complex(0, r.NormFloat64())
		}
	}
	a, _ := ndarray.FromSlice(d, shape...)
	if !cplx {
		return a.AsType(ndarray.Float64)
	}
	return a
}

func toArray(c cmat, cplx bool) *ndarray.Array {
	a, _ := ndarray.FromSlice(c.d, c.m, c.n)
	if !cplx {
		return a.AsType(ndarray.Float64)
	}
	return a
}

func eye(n int) cmat {
	c := cmat{n, n, make([]complex128, n*n)}
	for i := 0; i < n; i++ {
		c.d[i*n+i] = 1
	}
	return c
}

// within fails the test if got exceeds tol.
func within(t *testing.T, what string, got, tol float64) {
	t.Helper()
	if !(got <= tol) {
		t.Errorf("%s = %.3g, above %.3g", what, got, tol)
	}
}

func TestLargeIdentities(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for _, cplx := range []bool{false, true} {
		for _, n := range []int{100, 131} {
			a := randC(r, cplx, n, n)
			x := stack(a)[0]
			tol := idTol(n, "float64")
			// Inverse, solve, determinant.
			inv, err := Inv(a)
			if err != nil {
				t.Fatal(err)
			}
			within(t, "‖A·A⁻¹ - I‖", diffFro(mul(x, stack(inv)[0]), eye(n)), tol*1e3)
			// Cholesky of A·Aᴴ + nI.
			s := mul(x, adj(x))
			for i := 0; i < n; i++ {
				s.d[i*n+i] += complex(float64(n), 0)
			}
			l, err := Cholesky(toArray(s, cplx), Lower)
			if err != nil {
				t.Fatal(err)
			}
			lc := stack(l)[0]
			within(t, "‖LLᴴ - S‖/‖S‖", diffFro(mul(lc, adj(lc)), s)/fro(s), tol)
			// QR, tall and wide.
			for _, sh := range [][2]int{{n, n - 30}, {n - 30, n}} {
				b := randC(r, cplx, sh[0], sh[1])
				q, rr, err := QR(b, ModeComplete)
				if err != nil {
					t.Fatal(err)
				}
				qc, rc, bc := stack(q)[0], stack(rr)[0], stack(b)[0]
				within(t, "‖QᴴQ - I‖", orthErr(qc), tol)
				within(t, "‖A - QR‖/‖A‖", diffFro(mul(qc, rc), bc)/fro(bc), tol)
				u, sv, vh, err := SVD(b, true)
				if err != nil {
					t.Fatal(err)
				}
				c := fxCase{Outputs: []fxArray{{DType: "float64"}}, Inputs: []fxArray{{Shape: []int{sh[0], sh[1]}}}}
				if msg := checkSVD(c, b, []*ndarray.Array{u, sv, vh}); msg != "" {
					t.Errorf("SVD %v: %s", sh, msg)
				}
			}
			// Hermitian eigenproblem.
			h := toArray(hermitianOf(x, false), cplx)
			w, v, err := Eigh(h, Lower)
			if err != nil {
				t.Fatal(err)
			}
			c := fxCase{Args: map[string]any{"uplo": "L"}, Outputs: []fxArray{{}, {DType: "float64"}}, Inputs: []fxArray{{Shape: []int{n, n}}}}
			if msg := checkEigh(c, h, []*ndarray.Array{w, v}); msg != "" {
				t.Errorf("Eigh n=%d: %s", n, msg)
			}
			wv, _ := Eigvalsh(h, Lower)
			if d := maxDiff(vals(wv), vals(w)); d > tol*fro(x) {
				t.Errorf("Eigvalsh differs from Eigh by %.3g", d)
			}
			checkEigResidual(t, a)
		}
	}
}

func maxDiff(a, b []complex128) float64 {
	m := 0.0
	for i := range a {
		m = math.Max(m, cmplx.Abs(a[i]-b[i]))
	}
	return m
}

// checkEigResidual checks ‖Av - λv‖ ≤ tol·‖A‖ for every eigenpair, and
// that Eigvals agrees with Eig.
func checkEigResidual(t *testing.T, a *ndarray.Array) {
	t.Helper()
	w, v, err := Eig(a)
	if err != nil {
		t.Fatal(err)
	}
	x := stack(a)[0]
	n := x.n
	vc := stack(v)[0]
	av := mul(x, vc)
	ws := vals(w)
	for j := 0; j < n; j++ {
		res, nrm := 0.0, 0.0
		for i := 0; i < n; i++ {
			d := av.at(i, j) - vc.at(i, j)*ws[j]
			res += real(d)*real(d) + imag(d)*imag(d)
			nrm += real(vc.at(i, j))*real(vc.at(i, j)) + imag(vc.at(i, j))*imag(vc.at(i, j))
		}
		within(t, "eigenvector norm - 1", math.Abs(math.Sqrt(nrm)-1), idTol(n, "float64"))
		within(t, "‖Av - λv‖/‖A‖", math.Sqrt(res)/math.Max(fro(x), 1e-300), idTol(n, "float64"))
	}
	ev, err := Eigvals(a)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxDiff(vals(ev), ws); d > idTol(n, "float64")*fro(x) {
		t.Errorf("Eigvals differs from Eig by %.3g", d)
	}
}

// TestStackInParallel checks that a stack large enough to be spread over
// the processors gives each matrix's own result.
func TestStackInParallel(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4)) // also on a one-CPU runner
	r := rand.New(rand.NewPCG(13, 14))
	a := randC(r, false, 16, 40, 40)
	inv, err := Inv(a)
	if err != nil {
		t.Fatal(err)
	}
	got := vals(inv)
	for i, x := range stack(a) {
		one, _ := Inv(toArray(x, false))
		if d := maxDiff(got[i*1600:(i+1)*1600], vals(one)); d != 0 {
			t.Errorf("matrix %d differs by %g", i, d)
		}
	}
	// The error of the first failing matrix is reported.
	z := make([]float64, 16*40*40)
	zs, _ := ndarray.FromSlice(z, 16, 40, 40)
	if _, err := Inv(zs); !errors.Is(err, ErrSingular) {
		t.Errorf("Inv of zeros: %v", err)
	}
}

// TestNoConvergence lowers the iteration limits until every iteration fails
// and checks each function reports it.
func TestNoConvergence(t *testing.T) {
	defer func(f int) { iterFactor = f }(iterFactor)
	iterFactor = -1
	r := rand.New(rand.NewPCG(15, 16))
	for _, n := range []int{6, 40} {
		for _, cplx := range []bool{false, true} {
			a := randC(r, cplx, n, n)
			x := stack(a)[0]
			h := toArray(hermitianOf(x, false), cplx)
			b := randC(r, cplx, n)
			calls := map[string]func() error{
				"Eigh":       func() error { _, _, err := Eigh(h, Lower); return err },
				"Eigvalsh":   func() error { _, err := Eigvalsh(h, Lower); return err },
				"SVD":        func() error { _, _, _, err := SVD(a, false); return err },
				"SVDVals":    func() error { _, err := SVDVals(a); return err },
				"Eig":        func() error { _, _, err := Eig(a); return err },
				"Eigvals":    func() error { _, err := Eigvals(a); return err },
				"Lstsq":      func() error { _, err := Lstsq(a, b); return err },
				"Pinv":       func() error { _, err := Pinv(a); return err },
				"MatrixRank": func() error { _, err := MatrixRank(a); return err },
				"Cond":       func() error { _, err := Cond(a, Ord{}); return err },
				"Norm2":      func() error { _, err := Norm(a, P(2), nil, false); return err },
			}
			for name, f := range calls {
				if err := f(); !errors.Is(err, ErrNoConvergence) {
					t.Errorf("%s n=%d complex=%v: %v, want ErrNoConvergence", name, n, cplx, err)
				}
			}
		}
	}
}

// TestIterationLimit lets the tridiagonal QL/QR iteration start but stop
// short (one sweep per eigenvalue), in both directions.
func TestIterationLimit(t *testing.T) {
	defer func(f int) { iterFactor = f }(iterFactor)
	iterFactor = 1
	r := rand.New(rand.NewPCG(25, 26))
	for i := 0; i < 8; i++ {
		x := stack(randC(r, false, 7, 7))[0]
		h := hermitianOf(x, false)
		if i%2 == 1 {
			for j := range h.d {
				h.d[j] = -h.d[j] * complex(float64(1+j%7), 0)
			}
			h = hermitianOf(h, false)
		}
		if _, err := Eigvalsh(toArray(h, false), Lower); !errors.Is(err, ErrNoConvergence) {
			t.Errorf("matrix %d: %v", i, err)
		}
	}
}

// orthogonal is a seeded random n×n orthogonal matrix (the Q of a QR).
func orthogonal(r *rand.Rand, n int) cmat {
	q, _, _ := QR(randC(r, false, n, n), ModeReduced)
	return stack(q)[0]
}

func diagMat(d ...float64) cmat {
	n := len(d)
	c := cmat{n, n, make([]complex128, n*n)}
	for i, v := range d {
		c.d[i*n+i] = complex(v, 0)
	}
	return c
}

// TestDegenerateSpectra reaches the deflations of the divide-and-conquer
// solvers: repeated eigenvalues and singular values (close poles), and a
// zero matrix (zero updating vector).
func TestDegenerateSpectra(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 18))
	n := 40
	d := make([]float64, n)
	for i := range d {
		d[i] = float64(1 + i%3)
	}
	q := orthogonal(r, n)
	a := mul(mul(q, diagMat(d...)), adj(q))
	h := toArray(a, false)
	w, v, err := Eigh(h, Lower)
	if err != nil {
		t.Fatal(err)
	}
	c := fxCase{Args: map[string]any{"uplo": "L"}, Outputs: []fxArray{{}, {DType: "float64"}}, Inputs: []fxArray{{Shape: []int{n, n}}}}
	if msg := checkEigh(c, h, []*ndarray.Array{w, v}); msg != "" {
		t.Errorf("Eigh repeated: %s", msg)
	}
	// Singular values 1, 2, 3 repeated: U·diag·Vᵀ.
	p := orthogonal(r, n)
	b := toArray(mul(mul(q, diagMat(d...)), p), false)
	cs := fxCase{Outputs: []fxArray{{DType: "float64"}}, Inputs: []fxArray{{Shape: []int{n, n}}}}
	rd := toArray(mul(stack(randC(r, false, n, 7))[0], stack(randC(r, false, 7, n))[0]), false)
	for _, m := range []*ndarray.Array{b, h, rd, zerosArr(n, n), zerosArr(n+3, n)} {
		u, s, vh, err := SVD(m, false)
		if err != nil {
			t.Fatal(err)
		}
		cs.Inputs[0].Shape = m.Shape()
		if msg := checkSVD(cs, m, []*ndarray.Array{u, s, vh}); msg != "" {
			t.Errorf("SVD degenerate %v: %s", m.Shape(), msg)
		}
	}
	// A zero matrix: every eigenvalue 0, vectors the identity's.
	w, _, err = Eigh(zerosArr(n, n), Lower)
	if err != nil || maxAbs(vals(w)) != 0 {
		t.Errorf("Eigh of zeros: %v %v", vals(w), err)
	}
	// Tiny off-diagonal entries split the tridiagonal matrix.
	tri := diagMat(1, 2, 3, 4, 5)
	for i := 0; i < 4; i++ {
		tri.d[i*5+i+1], tri.d[(i+1)*5+i] = 1e-20, 1e-20
	}
	w, _, err = Eigh(toArray(tri, false), Lower)
	if err != nil || maxDiff(vals(w), []complex128{1, 2, 3, 4, 5}) > 1e-15 {
		t.Errorf("Eigh split: %v %v", vals(w), err)
	}
	// numpy returns NaN eigenvalues for a NaN input; the QL iteration here
	// reports it instead.
	nan := mul(mul(q, diagMat(d...)), adj(q))
	nan.d[0] = complex(math.NaN(), 0)
	if _, _, err := Eigh(toArray(nan, false), Lower); !errors.Is(err, ErrNoConvergence) {
		t.Errorf("Eigh of NaN: %v", err)
	}
}

func zerosArr(shape ...int) *ndarray.Array {
	a, _ := ndarray.Zeros(shape...)
	return a
}

// TestCholeskyFailsInBlocks: the leading minor that is not positive
// definite lies in the first, then the second half of the recursion.
func TestCholeskyFailsInBlocks(t *testing.T) {
	n := 100
	for _, bad := range []int{10, 80} {
		d := make([]float64, n)
		for i := range d {
			d[i] = 1
		}
		d[bad] = -1
		if _, err := Cholesky(toArray(diagMat(d...), false), Lower); !errors.Is(err, ErrNotPosDef) {
			t.Errorf("minor %d: %v", bad, err)
		}
	}
}

// TestParallelMatvec forces the parallel matrix-vector product of the
// tridiagonalization.
func TestParallelMatvec(t *testing.T) {
	defer func(v int) { matvecPar = v }(matvecPar)
	r := rand.New(rand.NewPCG(19, 20))
	a := randC(r, false, 120, 120)
	h := toArray(hermitianOf(stack(a)[0], false), false)
	want, _ := Eigvalsh(h, Lower)
	matvecPar = 1
	got, _ := Eigvalsh(h, Lower)
	if d := maxDiff(vals(got), vals(want)); d > 1e-12 {
		t.Errorf("parallel matvec changes the eigenvalues by %g", d)
	}
}

// TestEigBranches runs Eig on matrices that reach its rarer branches and
// checks the eigenpairs.
func TestEigBranches(t *testing.T) {
	mk := func(n int, f func(i, j int) complex128) cmat {
		c := cmat{n, n, make([]complex128, n*n)}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				c.d[i*n+j] = f(i, j)
			}
		}
		return c
	}
	cases := map[string]cmat{
		// Column 0 isolates an eigenvalue: balancing permutes it left.
		"isolated-column": mk(4, func(i, j int) complex128 {
			if j == 0 && i > 0 {
				return 0
			}
			return complex(float64(1+i+2*j), 0)
		}),
		// Cyclic permutations defeat the standard shifts.
		"cyclic": mk(12, func(i, j int) complex128 {
			if j == (i+1)%12 {
				return 1
			}
			return 0
		}),
		// A large Jordan block: the back-substitution overflows unless rescaled.
		"jordan": mk(60, func(i, j int) complex128 {
			switch j - i {
			case 0:
				return 2
			case 1:
				return 1
			}
			return 0
		}),
		// Two equal rotation blocks coupled: a defective complex pair.
		"defective-pair": mk(4, func(i, j int) complex128 {
			rot := [2][2]float64{{0, -1}, {1, 0}}
			switch {
			case i/2 == j/2:
				return complex(rot[i%2][j%2], 0)
			case j/2 == i/2+1 && i%2 == j%2:
				return 1
			}
			return 0
		}),
		// A zero diagonal.
		"zero-diagonal": mk(5, func(i, j int) complex128 {
			if i-j == 1 || j-i == 1 {
				return 1
			}
			return 0
		}),
		// Badly scaled: balancing scales by large powers of two.
		"scaled": mk(4, func(i, j int) complex128 {
			return complex(math.Pow(1e30, float64(i-j))*float64(1+(i*3+j)%5), 0)
		}),
	}
	r := rand.New(rand.NewPCG(21, 22))
	for _, n := range []int{8, 16} {
		cases[fmt.Sprintf("orthogonal-%d", n)] = orthogonal(r, n) // eigenvalues on the unit circle
		cases[fmt.Sprintf("grcar-%d", n)] = mk(n, func(i, j int) complex128 {
			if j == i-1 {
				return -1
			}
			if j >= i && j <= i+3 {
				return 1
			}
			return 0
		})
		cases[fmt.Sprintf("jordan-1e10-%d", n)] = mk(n, func(i, j int) complex128 {
			switch j - i {
			case 0:
				return 3
			case 1:
				return 1e10
			}
			return 0
		})
		// Coupled rotation blocks: a defective complex pair of high order.
		cases[fmt.Sprintf("rotations-%d", n)] = mk(2*n, func(i, j int) complex128 {
			rot := [2][2]float64{{0, -1}, {1, 0}}
			switch {
			case i/2 == j/2:
				return complex(rot[i%2][j%2], 0)
			case j/2 == i/2+1 && i%2 == j%2:
				return 1e5
			}
			return 0
		})
	}
	for _, n := range []int{6, 30} {
		cases[fmt.Sprintf("frank-%d", n)] = mk(n, func(i, j int) complex128 {
			if j >= i-1 {
				return complex(float64(n-max(i, j)), 0)
			}
			return 0
		})
		cases[fmt.Sprintf("rotations-%d", n)] = mk(2*n, func(i, j int) complex128 {
			rot := [2][2]float64{{0, -1}, {1, 0}}
			switch {
			case i/2 == j/2:
				return complex(rot[i%2][j%2], 0)
			case j/2 == i/2+1 && i%2 == j%2:
				return 1e5
			}
			return 0
		})
		cases[fmt.Sprintf("jordan-1e10-%d", n)] = mk(n, func(i, j int) complex128 {
			switch j - i {
			case 0:
				return 3
			case 1:
				return 1e10
			}
			return 0
		})
		cases[fmt.Sprintf("binary-%d", n)] = mk(n, func(i, j int) complex128 { return complex(float64(r.IntN(2)), 0) })
	}
	// Columns 2 and 0 isolate eigenvalues.
	cases["isolated-columns"] = mk(5, func(i, j int) complex128 {
		if (j == 2 && i != 2) || (j == 0 && i != 0) {
			return 0
		}
		return complex(float64(1+i+2*j), 0)
	})
	// Scaling at the limits of the exponent range.
	cases["extreme-scaling"] = cmat{2, 2, []complex128{1, 1e300, 1e-300, 1}}
	cases["extreme-scaling-t"] = cmat{2, 2, []complex128{1, 1e-300, 1e300, 1}}
	// A chain whose balancing would need scale factors beyond the exponent
	// range: balancing stops at its limits.
	cases["scaling-chain"] = cmat{3, 3, []complex128{1, 1e300, 0, 1e-300, 1, 1e300, 0, 1e-300, 1}}
	cases["scaling-chain-t"] = cmat{3, 3, []complex128{1, 1e-300, 0, 1e300, 1, 1e-300, 0, 1e300, 1}}
	// Equal pairs far from the origin: λ - T is below smin.
	cases["shifted-pairs"] = mk(8, func(i, j int) complex128 {
		rot := [2][2]float64{{1e20, -1}, {1, 1e20}}
		switch {
		case i/2 == j/2:
			return complex(rot[i%2][j%2], 0)
		case j/2 == i/2+1 && i%2 == j%2:
			return 1
		}
		return 0
	})
	for name, c := range cases {
		for _, cplx := range []bool{false, true} {
			m := c
			if cplx {
				m = cmat{c.m, c.n, append([]complex128(nil), c.d...)}
				m.d[0] += 1i
			}
			t.Run(name, func(t *testing.T) { checkEigResidual(t, toArray(m, cplx)) })
		}
	}
}

// TestExceptionalShifts makes every iteration use an exceptional shift:
// the QR iterations must then still converge to correct eigenpairs, or
// say they did not.
func TestExceptionalShifts(t *testing.T) {
	defer func(k int) { kexsh = k }(kexsh)
	kexsh = 1
	r := rand.New(rand.NewPCG(23, 24))
	for _, cplx := range []bool{false, true} {
		for _, n := range []int{3, 8} {
			a := randC(r, cplx, n, n)
			if _, _, err := Eig(a); err != nil {
				if !errors.Is(err, ErrNoConvergence) {
					t.Errorf("n=%d complex=%v: %v", n, cplx, err)
				}
				continue
			}
			checkEigResidual(t, a)
		}
	}
}
