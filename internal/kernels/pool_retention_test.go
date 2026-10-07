package kernels

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"
	"weak"
)

// TestPoolDoesNotRetainLastJob: once parallelFor returns, nothing in the pool
// may keep the job's data alive. The last job used to stay in pool.job, its
// closure holding the operands, so the arrays of the last parallel operation
// could not be collected until another one replaced it.
func TestPoolDoesNotRetainLastJob(t *testing.T) {
	big := make([]float64, 1<<20)
	ref := weak.Make(&big[0])
	fill(big) // the closure captures fill's parameter, which nothing clears
	big = nil
	for i := 0; i < 5 && ref.Value() != nil; i++ {
		runtime.GC()
	}
	if ref.Value() != nil {
		t.Fatal("the last parallelFor job keeps its data alive after returning")
	}
}

func fill(a []float64) {
	parallelFor(len(a), 4, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			a[i] = 1
		}
	})
}

// TestPoolCallerPanicStopsTheHelpers: when body panics in the caller, the
// panic propagates only after the helpers have stopped, and no block starts
// afterwards; the pool then serves the next call.
func TestPoolCallerPanicStopsTheHelpers(t *testing.T) {
	var after atomic.Int64
	var stopped atomic.Bool
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("the panic did not propagate")
			}
			stopped.Store(true)
		}()
		parallelFor(64, 4, func(lo, hi int) {
			if lo == 0 {
				panic("boom") // block 0 is the first the caller claims
			}
			time.Sleep(time.Millisecond)
			if stopped.Load() {
				after.Add(1)
			}
		})
	}()
	time.Sleep(20 * time.Millisecond)
	if n := after.Load(); n != 0 {
		t.Fatalf("%d blocks ran after the panic propagated", n)
	}
	checkCover(t, 1000, 4)
}
