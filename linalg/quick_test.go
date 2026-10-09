package linalg

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

func randMat(r *rand.Rand, m, n int) *ndarray.Array {
	d := make([]float64, m*n)
	for i := range d {
		d[i] = r.NormFloat64()
	}
	a, _ := ndarray.FromSlice(d, m, n)
	return a
}

func TestQuick(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{1, 5, 33, 100, 300} {
		a := randMat(r, n, n)
		inv, err := Inv(a)
		if err != nil {
			t.Fatal(err)
		}
		p, _ := a.MatMul(inv)
		d, _ := ndarray.Data[float64](p)
		e := 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				w := 0.0
				if i == j {
					w = 1
				}
				e = math.Max(e, math.Abs(d[i*n+j]-w))
			}
		}
		at := a.Transpose()
		spd, _ := a.MatMul(at)
		l, err := Cholesky(spd, Lower)
		if err != nil {
			t.Fatal(err)
		}
		llt, _ := l.MatMul(l.Transpose())
		diff, _ := llt.Sub(spd)
		ld, _ := ndarray.Data[float64](diff)
		ce := 0.0
		for _, v := range ld {
			ce = math.Max(ce, math.Abs(v))
		}
		q, rr, _ := QR(a, ModeReduced)
		qr, _ := q.MatMul(rr)
		diff, _ = qr.Sub(a)
		ld, _ = ndarray.Data[float64](diff)
		qe := 0.0
		for _, v := range ld {
			qe = math.Max(qe, math.Abs(v))
		}
		t.Logf("n=%d inv err %g chol err %g qr err %g", n, e, ce, qe)
	}
}

func maxAbsDiff(a, b *ndarray.Array) float64 {
	d, _ := a.Sub(b)
	d = d.Abs()
	m, _ := d.Max()
	return m
}

func TestQuickEigh(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 3))
	for _, n := range []int{1, 2, 5, 33, 100, 300} {
		a := randMat(r, n, n)
		s, _ := a.Add(a.Transpose())
		w, v, err := Eigh(s, Lower)
		if err != nil {
			t.Fatal(err)
		}
		av, _ := s.MatMul(v)
		wd, _ := ndarray.Data[float64](w)
		vd, _ := ndarray.Data[float64](v)
		lv := make([]float64, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				lv[i*n+j] = vd[i*n+j] * wd[j]
			}
		}
		lva, _ := ndarray.FromSlice(lv, n, n)
		vtv, _ := v.Transpose().MatMul(v)
		id, _ := ndarray.Eye(n, n, 0)
		t.Logf("n=%d resid %g orth %g", n, maxAbsDiff(av, lva), maxAbsDiff(vtv, id))
	}
}

func TestQuickSVD(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 4))
	for _, sh := range [][2]int{{1, 1}, {3, 2}, {2, 3}, {5, 5}, {40, 33}, {33, 40}, {200, 150}} {
		m, n := sh[0], sh[1]
		a := randMat(r, m, n)
		for _, full := range []bool{false, true} {
			u, s, vh, err := SVD(a, full)
			if err != nil {
				t.Fatal(err)
			}
			k := min(m, n)
			sd, _ := ndarray.Data[float64](s)
			ud, _ := ndarray.Data[float64](u)
			cu := u.Shape()[1]
			us := make([]float64, m*k)
			for i := 0; i < m; i++ {
				for j := 0; j < k; j++ {
					us[i*k+j] = ud[i*cu+j] * sd[j]
				}
			}
			usa, _ := ndarray.FromSlice(us, m, k)
			vk, _ := vh.Slice(ndarray.To(k), ndarray.All())
			rec, _ := usa.MatMul(vk)
			utu, _ := u.Transpose().MatMul(u)
			vvt, _ := vh.MatMul(vh.Transpose())
			i1, _ := ndarray.Eye(cu, cu, 0)
			i2, _ := ndarray.Eye(vh.Shape()[0], vh.Shape()[0], 0)
			t.Logf("%dx%d full=%v rec %g orthU %g orthV %g s0 %g", m, n, full, maxAbsDiff(rec, a), maxAbsDiff(utu, i1), maxAbsDiff(vvt, i2), sd[0])
		}
	}
}

func eigResid(t *testing.T, a *ndarray.Array) {
	t.Helper()
	n := a.Shape()[0]
	w, v, err := Eig(a)
	if err != nil {
		t.Fatal(err)
	}
	ac := a.AsType(ndarray.Complex128)
	av, _ := ac.MatMul(v)
	wd, _ := ndarray.Data[complex128](w)
	vd, _ := ndarray.Data[complex128](v)
	avd, _ := ndarray.Data[complex128](av)
	e := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d := avd[i*n+j] - vd[i*n+j]*wd[j]
			e = math.Max(e, math.Hypot(real(d), imag(d)))
		}
	}
	nrm := 0.0
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += real(vd[i*n+j])*real(vd[i*n+j]) + imag(vd[i*n+j])*imag(vd[i*n+j])
		}
		nrm = math.Max(nrm, math.Abs(s-1))
	}
	t.Logf("n=%d %v resid %g norm %g", n, a.DType(), e, nrm)
}

func TestQuickEig(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 5))
	for _, n := range []int{1, 2, 3, 5, 33, 100, 250} {
		a := randMat(r, n, n)
		eigResid(t, a)
		b := randMat(r, n, n)
		bi, _ := b.AsType(ndarray.Complex128).Mul(ndarray.Scalar(1i))
		c, _ := a.AsType(ndarray.Complex128).Add(bi)
		eigResid(t, c)
	}
	j, _ := ndarray.FromSlice([]float64{1, 1, 0, 1}, 2, 2)
	eigResid(t, j)
	p, _ := ndarray.FromSlice([]float64{0, 1, 0, 0, 0, 1, 1, 0, 0}, 3, 3)
	eigResid(t, p)
}
