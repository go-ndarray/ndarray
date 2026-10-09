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

