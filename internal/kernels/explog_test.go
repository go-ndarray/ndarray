package kernels

import (
	"math"
	"testing"
)

// The exp/log kernels are go-simd/floats' (proofs and kernel tests live there);
// what stays here is the parallel split and Log10: every P form gives the serial
// bits, below and above ParThreshold.
func TestExpLogParallelMatchesSerial(t *testing.T) {
	src := randVec(5000, 3)
	for i := range src {
		src[i] = math.Abs(src[i]) * 40
	}
	src32 := make([]float32, len(src))
	for i, v := range src {
		src32[i] = float32(v)
	}
	run := func(name string, serial, par func(dst, src []float64)) {
		want, got := make([]float64, len(src)), make([]float64, len(src))
		serial(want, src)
		withThresholds(64, 1<<14, func() { par(got, src) })
		eqSlice(t, got, want)
		par(got[:10], src[:10]) // below the threshold
		eqSlice(t, got[:10], want[:10])
		_ = name
	}
	run("exp", Exp, ExpP)
	run("log", Log, LogP)
	run("log10", Log10, Log10P)
	for _, c := range []struct {
		name        string
		serial, par func(dst, src []float32)
	}{{"exp32", Exp32, Exp32P}, {"log32", Log32, Log32P}} {
		want, got := make([]float32, len(src32)), make([]float32, len(src32))
		c.serial(want, src32)
		withThresholds(64, 1<<14, func() { c.par(got, src32) })
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%sP [%d]: %v, serial %v", c.name, i, got[i], want[i])
			}
		}
		c.par(got[:10], src32[:10])
	}
	// Log10 is log(x)*(1/ln10).
	d := make([]float64, 1)
	Log10(d, []float64{1000})
	if math.Abs(d[0]-3) > 1e-15 {
		t.Errorf("Log10(1000) = %v", d[0])
	}
}
