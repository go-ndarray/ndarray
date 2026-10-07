package kernels

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestParallelStress is a differential stress test of the parallel paths and
// the helper pool behind them. Several goroutines run parallel kernels at once
// (so the pool is contended and some calls take the fork fallback) while
// another keeps changing GOMAXPROCS, and every result is compared with the
// serial kernel. The data are small integers, so every sum, product and dot is
// exact and the comparison is equality; Exp runs the same per-element code in
// both, so it is bit-identical too.
func TestParallelStress(t *testing.T) {
	dur := 1500 * time.Millisecond
	if testing.Short() {
		dur = 200 * time.Millisecond
	}
	op, og, omv := ParThreshold, GemmThreshold, matVecThreshold
	ParThreshold, GemmThreshold, matVecThreshold = 64, 64, 64 // parallel from tiny sizes
	defer func() { ParThreshold, GemmThreshold, matVecThreshold = op, og, omv }()
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))

	stop := make(chan struct{})
	var procs sync.WaitGroup
	procs.Add(1)
	go func() { // churn GOMAXPROCS under the callers
		defer procs.Done()
		r := rand.New(rand.NewSource(1))
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GOMAXPROCS(1 + r.Intn(16))
				time.Sleep(time.Duration(r.Intn(300)) * time.Microsecond)
			}
		}
	}()

	var calls atomic.Int64
	var wg sync.WaitGroup
	deadline := time.Now().Add(dur)
	for c := 0; c < 6; c++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			r := rand.New(rand.NewSource(seed))
			ints := func(n int) []float64 {
				a := make([]float64, n)
				for i := range a {
					a[i] = float64(r.Intn(7) - 3)
				}
				return a
			}
			for time.Now().Before(deadline) {
				n := 1 + r.Intn(5000)
				a, b := ints(n), ints(n)
				got, want := make([]float64, n), make([]float64, n)
				switch r.Intn(9) {
				case 0:
					AddP(got, a, b)
					Add(want, a, b)
				case 1:
					MulP(got, a, b)
					Mul(want, a, b)
				case 2:
					got[0], want[0] = SumP(a), Sum(a)
				case 3:
					got[0], want[0] = MaxP(a), Max(a)
				case 4:
					got[0], want[0] = Dot1DP(a, b), Dot1D(a, b)
				case 5:
					m := 1 + r.Intn(64)
					k := n / m
					if k == 0 {
						continue
					}
					got, want = got[:m], want[:m]
					MatVecP(got, a[:m*k], b[:k], m, k)
					for i := 0; i < m; i++ {
						want[i] = Dot1D(a[i*k:i*k+k], b[:k])
					}
				case 6:
					outer, inner := 1+r.Intn(8), 1+r.Intn(700)
					axis := n / (outer * inner)
					if axis == 0 {
						continue
					}
					src := a[:outer*axis*inner]
					got, want = make([]float64, outer*inner), make([]float64, outer*inner)
					RunAxisP(SumAxis, got, src, outer, axis, inner)
					SumAxis(want, src, outer, axis, inner, 0, inner)
				case 7:
					ExpP(got, a)
					Exp(want, a)
				default:
					SqrtP(got, b)
					sqrtSIMD(want, b)
				}
				for i := range want {
					if got[i] != want[i] && !(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
						t.Errorf("seed %d n=%d: [%d] parallel %v, serial %v", seed, n, i, got[i], want[i])
						return
					}
				}
				calls.Add(1)
			}
		}(int64(c + 2))
	}
	// A lost block hangs the call rather than corrupting it: fail by name.
	finished := make(chan struct{})
	go func() { wg.Wait(); close(finished) }()
	select {
	case <-finished:
	case <-time.After(dur + 10*time.Second):
		t.Fatal("a parallel call never returned: a block was lost")
	}
	close(stop)
	procs.Wait()
	t.Logf("%d parallel calls checked", calls.Load())
}
