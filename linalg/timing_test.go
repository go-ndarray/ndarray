package linalg

import (
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
)

func TestTimingEigh(t *testing.T) {
	n := 1024
	a := benchMatrix(n, "sym")
	for rep := 0; rep < 3; rep++ {
		x, _ := ndarray.Data[float64](a.Copy())
		t0 := time.Now()
		hetrd(n, x, false)
		t1 := time.Now()
		x, _ = ndarray.Data[float64](a.Copy())
		hetrd(n, x, true)
		t2 := time.Now()
		t.Logf("hetrd %v +q %v", t1.Sub(t0), t2.Sub(t1))
	}
}

func BenchmarkHetrd(b *testing.B) {
	n := 1024
	a := benchMatrix(n, "sym")
	for b.Loop() {
		x, _ := ndarray.Data[float64](a.Copy())
		hetrd(n, x, false)
	}
}


func TestTimingSVD(t *testing.T) {
	n := 1024
	a := benchMatrix(n, "")
	for rep := 0; rep < 2; rep++ {
		x, _ := ndarray.Data[float64](a.Copy())
		t0 := time.Now()
		bd := gebrd(n, n, x, true)
		d, e, tauq, pv, taup := bd.d, bd.e, bd.tauq, bd.pv, bd.taup
		t1 := time.Now()
		orgqr(n, n, n, x, n, tauq)
		orgqr(n-1, n-1, n-1, pv, n-1, taup)
		t2 := time.Now()
		bdsdc(n, d, e)
		t3 := time.Now()
		t.Logf("gebd2 %v orgqr×2 %v bdsdc %v", t1.Sub(t0), t2.Sub(t1), t3.Sub(t2))
	}
}
