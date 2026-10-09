package linalg

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestQuickBdsdc(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{5, 26, 60, 200, 513} {
		d, e := make([]float64, n), make([]float64, n-1)
		for i := range d {
			d[i] = r.NormFloat64()
		}
		for i := range e {
			e[i] = r.NormFloat64()
		}
		d0, e0 := append([]float64(nil), d...), append([]float64(nil), e...)
		u, vt, ok := bdsdc(n, d, e)
		if !ok {
			t.Fatal("no conv")
		}
		// residual B - U S VT and orthogonality
		res, orth := 0.0, 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				b := 0.0
				if i == j {
					b = d0[i]
				} else if j == i+1 {
					b = e0[i]
				}
				s := 0.0
				for k := 0; k < n; k++ {
					s += u[i*n+k] * d[k] * vt[k*n+j]
				}
				res = math.Max(res, math.Abs(s-b))
				uu, vv := 0.0, 0.0
				for k := 0; k < n; k++ {
					uu += u[k*n+i] * u[k*n+j]
					vv += vt[i*n+k] * vt[j*n+k]
				}
				if i == j {
					uu--
					vv--
				}
				orth = math.Max(orth, math.Max(math.Abs(uu), math.Abs(vv)))
			}
		}
		d2, e2 := append([]float64(nil), d0...), append([]float64(nil), e0...)
		bdsqr[float64](n, d2, e2, nil, 0, nil, 0)
		sd := 0.0
		for i := range d2 {
			sd = math.Max(sd, math.Abs(d2[i]-d[i]))
		}
		t.Logf("n=%d resid %.3g orth %.3g sv diff %.3g", n, res, orth, sd)
	}
}
