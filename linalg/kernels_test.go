package linalg

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Unit tests of the scalar kernels, on the inputs that reach their special
// cases, each checked through its defining property.

func close(a, b, scale float64) bool { return math.Abs(a-b) <= 1e-14*scale }

func TestLaev2(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	cases := [][3]float64{{1, 0, 1}, {1, 1, -1}, {2, 0, 1}, {-3, 1, 2}, {1, 2, 1}}
	for i := 0; i < 20; i++ {
		cases = append(cases, [3]float64{r.NormFloat64(), r.NormFloat64(), r.NormFloat64()})
	}
	for _, c := range cases {
		a, b, cc := c[0], c[1], c[2]
		rt1, rt2, cs, sn := laev2(a, b, cc)
		sc := math.Abs(a) + math.Abs(b) + math.Abs(cc)
		if !close(a*cs+b*sn, rt1*cs, sc) || !close(b*cs+cc*sn, rt1*sn, sc) || !close(rt1+rt2, a+cc, sc) ||
			math.Abs(rt1) < math.Abs(rt2) || !close(cs*cs+sn*sn, 1, 1) {
			t.Errorf("laev2%v = %v %v %v %v", c, rt1, rt2, cs, sn)
		}
	}
}

// sv2 is the singular values of [f g; 0 h], from the eigenvalues of AᵀA.
func sv2(f, g, h float64) (float64, float64) {
	tr := f*f + g*g + h*h
	det := math.Abs(f * h)
	big := math.Sqrt((tr + math.Sqrt(math.Max(tr*tr-4*det*det, 0))) / 2)
	if big == 0 {
		return 0, 0
	}
	return det / big, big
}

func TestLas2Lasv2(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	cases := [][3]float64{
		{0, 0, 0}, {0, 3, 0}, {0, 3, 4}, {4, 3, 0}, {1, 1e20, 1}, {1e-200, 1e200, 1e-200},
		{1, 1e-300, 1}, {2, 1e-300, 1}, {1, 0, 2}, {2, 0, 1}, {1e-17, 1, 2}, {2, 1, 1e-17},
		{3, 1e17, 2}, {0.5, 1e17, 0.25}, {-2, 5, 3}, {2, -5, -3},
	}
	for i := 0; i < 30; i++ {
		cases = append(cases, [3]float64{r.NormFloat64(), r.NormFloat64() * 10, r.NormFloat64()})
	}
	for _, c := range cases {
		f, g, h := c[0], c[1], c[2]
		lo, hi := sv2(f, g, h)
		smin, smax := las2(f, g, h)
		if !close(smin, lo, hi) || !close(smax, hi, hi) {
			t.Errorf("las2%v = %v %v, want %v %v", c, smin, smax, lo, hi)
		}
		ssmin, ssmax, snr, csr, snl, csl := lasv2(f, g, h)
		// [csl snl; -snl csl]·[f g; 0 h]·[csr -snr; snr csr] = diag(ssmax, ssmin)
		a11 := csl*(f*csr+g*snr) + snl*(h*snr)
		a12 := csl*(-f*snr+g*csr) + snl*(h*csr)
		a21 := -snl*(f*csr+g*snr) + csl*(h*snr)
		a22 := -snl*(-f*snr+g*csr) + csl*(h*csr)
		if !close(a11, ssmax, hi) || !close(a22, ssmin, hi) || !close(a12, 0, hi) || !close(a21, 0, hi) ||
			!close(math.Abs(ssmax), hi, hi) || !close(math.Abs(ssmin), lo, hi) {
			t.Errorf("lasv2%v = %v %v: %v %v %v %v", c, ssmin, ssmax, a11, a12, a21, a22)
		}
	}
}

func TestLanv2(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	cases := [][4]float64{
		{1, 2, 0, 3}, {1, 0, 2, 3}, {1, 2, -3, 1}, {1, 1, 1e-17, 1}, {1, 1e-17, 1, 1},
		{1e200, 3e200, -2e200, 1.0000001e200}, {1e-200, 3e-200, -2e-200, 1e-200}, {1e-200, 3e-200, -2e-200, 1.5e-200}, {1, 0, -1e-17, 1},
		{1, 1, 1, 1}, {2, -1, 1, 2}, {0, 1, -1, 0}, {1, 2, 2, 1},
	}
	for i := 0; i < 50; i++ {
		cases = append(cases, [4]float64{r.NormFloat64(), r.NormFloat64(), r.NormFloat64(), r.NormFloat64()})
	}
	for _, c := range cases {
		a, b, cc, d := c[0], c[1], c[2], c[3]
		aa, bb, c2, dd, rt1r, rt1i, rt2r, rt2i, cs, sn := lanv2(a, b, cc, d)
		sc := math.Abs(a) + math.Abs(b) + math.Abs(cc) + math.Abs(d)
		// [a b; c d] = [cs -sn; sn cs]·[aa bb; cc dd]·[cs sn; -sn cs]
		m11, m12 := cs*aa-sn*c2, cs*bb-sn*dd
		m21, m22 := sn*aa+cs*c2, sn*bb+cs*dd
		r11, r12 := m11*cs-m12*sn, m11*sn+m12*cs
		r21, r22 := m21*cs-m22*sn, m21*sn+m22*cs
		std := c2 == 0 || (aa == dd && math.Signbit(bb) != math.Signbit(c2))
		eig := rt1r == aa && rt2r == dd && (c2 == 0) == (rt1i == 0) && rt2i == -rt1i
		if !close(r11, a, sc) || !close(r12, b, sc) || !close(r21, cc, sc) || !close(r22, d, sc) || !std || !eig {
			t.Errorf("lanv2%v = %v %v %v %v", c, aa, bb, c2, dd)
		}
	}
}

func TestLartgExtremes(t *testing.T) {
	for _, c := range [][2]float64{{1e300, 1e300}, {1e-300, 1e-300}, {3, 4}, {0, -2}, {5, 0}, {-1e300, 1e-300}} {
		f, g := c[0], c[1]
		cs, sn, r := lartg(f, g)
		sc := math.Max(math.Abs(f), math.Abs(g))
		if !close(cs*f+sn*g, r, sc) || !close(-sn*f+cs*g, 0, sc) || !close(cs*cs+sn*sn, 1, 1) {
			t.Errorf("lartg%v = %v %v %v", c, cs, sn, r)
		}
	}
}

func TestNrm2Lapy3(t *testing.T) {
	if lapy3(0, 0, 0) != 0 || lapy3(3, 0, 4) != 5 {
		t.Error("lapy3")
	}
	for _, c := range []struct {
		x    []float64
		want float64
	}{
		{[]float64{1e200, 1e200}, 1e200 * math.Sqrt2},
		{[]float64{1e-200, 1e-200}, 1e-200 * math.Sqrt2},
		{[]float64{0, 0}, 0},
		{[]float64{3, 4}, 5},
		{[]float64{math.Inf(1), 1}, math.Inf(1)},
	} {
		if got := nrm2(len(c.x), c.x, 1); !close(got, c.want, c.want) && got != c.want {
			t.Errorf("nrm2%v = %v", c.x, got)
		}
	}
	if !math.IsNaN(nrm2(2, []float64{math.NaN(), 1}, 1)) {
		t.Error("nrm2 of NaN")
	}
	if got := nrm2(1, []complex128{3e200 + 4e200i}, 1); !close(got, 5e200, 5e200) {
		t.Errorf("complex nrm2 = %v", got)
	}
}

func TestQuadRoot(t *testing.T) {
	// c = 0: one root of the linear equation.
	if x := quadRoot(0, 1, 1, 0, 2, 0, 2); !close(x, 1, 1) {
		t.Errorf("linear root %v", x)
	}
	// No real root.
	if x := quadRoot(1, 1, -1, 0, 1, 0, 1); !math.IsNaN(x) {
		t.Errorf("complex roots gave %v", x)
	}
	// The root in the bracket, from either formula.
	for _, c := range [][5]float64{{1, 1, 1, 0, 3}, {-1, 1, 1, 0, 3}} {
		x := quadRoot(c[0], c[1], c[2], c[3], c[4], c[3], c[4])
		if f := c[0] + c[1]/(c[3]-x) + c[2]/(c[4]-x); !(x > c[3] && x < c[4]) || math.Abs(f) > 1e-12 {
			t.Errorf("quadRoot%v = %v (f = %v)", c, x, f)
		}
	}
}

func TestSolve2(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for i := 0; i < 20; i++ {
		var m [4]complex128
		for j := range m {
			m[j] = complex(r.NormFloat64(), r.NormFloat64())
		}
		r1, r2 := complex(r.NormFloat64(), 0), complex(0, r.NormFloat64())
		y1, y2, s := solve2(m[0], m[1], m[2], m[3], r1, r2, 1e-300, 1e300)
		if s != 1 || cmplx.Abs(m[0]*y1+m[1]*y2-r1) > 1e-12 || cmplx.Abs(m[2]*y1+m[3]*y2-r2) > 1e-12 {
			t.Errorf("solve2: %v %v", y1, y2)
		}
	}
	// A matrix below smin is perturbed to smin·I, and a huge solution scaled.
	y1, y2, s := solve2(1e-30, 0, 0, 1e-30, 1e3, 1e3, 1e-10, 1e5)
	if y1 != y2 || s >= 1 || cmplx.Abs(y1) > 1e10 {
		t.Errorf("perturbed solve2: %v %v %v", y1, y2, s)
	}
}

// TestBdsqrGraded runs the bidiagonal QR on graded matrices, where it
// chases the bulge both ways with zero shifts, and checks B = U·S·Vᵀ to
// within rounding of ‖B‖.
func TestBdsqrGraded(t *testing.T) {
	type bidi struct{ d, e []float64 }
	var cases []bidi
	for _, dir := range []float64{1, -1} {
		n := 12
		d, e := make([]float64, n), make([]float64, n-1)
		for i := range d {
			d[i] = math.Pow(10, dir*float64(i-n/2)*2)
		}
		for i := range e {
			e[i] = math.Pow(10, dir*float64(i-n/2)*2) * 0.5
		}
		cases = append(cases, bidi{d, e})
	}
	// Found by a search: a zero-shift sweep from the bottom deflates at the top.
	cases = append(cases, bidi{
		[]float64{1.087, 14326.298, 1.6734395e+07, 6.8290222558999234e+13, 6.1163180038118536e+16, 5.214581644332932e+21, 1.2978735465153189e+25},
		[]float64{0.156, 833.053, 704284.894, 8.620413772825534e+21, 1.631282114430647e+12, 4.406843216880124e+22},
	})
	for _, c := range cases {
		n := len(c.d)
		d, e := append([]float64(nil), c.d...), append([]float64(nil), c.e...)
		vt, ut := make([]float64, n*n), make([]float64, n*n)
		for i := 0; i < n; i++ {
			vt[i*n+i], ut[i*n+i] = 1, 1
		}
		if !bdsqr(n, d, e, vt, n, ut, n) {
			t.Fatal("bdsqr did not converge")
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				s := 0.0
				for k := 0; k < n; k++ {
					s += ut[k*n+i] * d[k] * vt[k*n+j]
				}
				want := 0.0
				if i == j {
					want = c.d[i]
				} else if j == i+1 {
					want = c.e[i]
				}
				if math.Abs(s-want) > 1e-13*d[0] {
					t.Errorf("B[%d,%d] = %v, want %v", i, j, s, want)
				}
			}
		}
	}
}

func TestNegligibleShift(t *testing.T) {
	if negligibleShift(1e-9, 1) != 0 || negligibleShift(1e-7, 1) != 1e-7 || negligibleShift(1, 0) != 1 {
		t.Error("negligibleShift")
	}
}

func TestSmallAPI(t *testing.T) {
	for o, s := range map[Ord]string{{}: "None", Fro: "fro", Nuc: "nuc", P(2): "2"} {
		if o.String() != s {
			t.Errorf("Ord %q", o.String())
		}
	}
	a, _ := ndarray.Ones(3, 3)
	if _, _, err := QR(a, ModeRaw+1); !errors.Is(err, ErrArgs) {
		t.Errorf("QR bad mode: %v", err)
	}
	v, _ := ndarray.Ones(3)
	if _, err := SVDVals(v); !errors.Is(err, ErrShape) {
		t.Errorf("SVDVals 1-D: %v", err)
	}
}

// TestBdsdcZeroDiagonal: zero diagonal entries give the halves zero
// singular values, which the merge must keep apart from the arrow's own
// zero pole.
func TestBdsdcZeroDiagonal(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	n := 60
	for _, zeros := range [][]int{{5}, {5, 17}, {29, 30, 45}, {0, 59}} {
		d, e := make([]float64, n), make([]float64, n-1)
		for i := range d {
			d[i] = r.NormFloat64()
		}
		for i := range e {
			e[i] = r.NormFloat64()
		}
		for _, z := range zeros {
			d[z] = 0
		}
		d0, e0 := append([]float64(nil), d...), append([]float64(nil), e...)
		u, vt, ok := bdsdc(n, d, e)
		if !ok {
			t.Fatal("bdsdc did not converge")
		}
		worst := 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				s := 0.0
				for k := 0; k < n; k++ {
					s += u[i*n+k] * d[k] * vt[k*n+j]
				}
				want := 0.0
				if i == j {
					want = d0[i]
				} else if j == i+1 {
					want = e0[i]
				}
				worst = math.Max(worst, math.Abs(s-want))
			}
		}
		uc, vc := cmat{n, n, make([]complex128, n*n)}, cmat{n, n, make([]complex128, n*n)}
		for i := range u {
			uc.d[i], vc.d[i] = complex(u[i], 0), complex(vt[i], 0)
		}
		if worst > 1e-13*d[0] || orthErr(uc) > 1e-12 || orthErr(adj(vc)) > 1e-12 {
			t.Errorf("zeros %v: ‖B - USVᵀ‖∞ = %.3g, orth %.3g %.3g", zeros, worst, orthErr(uc), orthErr(adj(vc)))
		}
	}
}
