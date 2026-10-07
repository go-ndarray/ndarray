package kernels

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// checkCover runs parallelFor(n, w) and asserts every index is visited once.
func checkCover(t *testing.T, n, w int) {
	t.Helper()
	seen := make([]atomic.Int32, n)
	parallelFor(n, w, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			seen[i].Add(1)
		}
	})
	for i := range seen {
		if c := seen[i].Load(); c != 1 {
			t.Fatalf("n=%d w=%d: index %d visited %d times", n, w, i, c)
		}
	}
}

// TestPoolParksAndWakes: with no spin window every helper parks between jobs
// and must be woken for the next; with a long one they stay awake. Both must
// cover every index, back to back, for worker counts below and above the
// number of helpers already started.
func TestPoolParksAndWakes(t *testing.T) {
	saved := spinWindow.Load()
	defer spinWindow.Store(saved)
	for _, win := range []time.Duration{0, time.Millisecond} {
		spinWindow.Store(int64(win))
		for rep := 0; rep < 50; rep++ {
			for _, w := range []int{2, 7, 3, 12} {
				checkCover(t, 1000+rep, w)
			}
			if win == 0 {
				time.Sleep(50 * time.Microsecond) // let helpers reach the park
			}
		}
	}
}

// TestPoolConcurrentCallers: callers that find the pool busy fork their own
// goroutines; every call still covers its range exactly once.
func TestPoolConcurrentCallers(t *testing.T) {
	var wg sync.WaitGroup
	for c := 0; c < 8; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rep := 0; rep < 100; rep++ {
				checkCover(t, 500+rep, 4)
			}
		}()
	}
	wg.Wait()
}

// TestPoolMoreHelpersThanWorkers: after a wide job, a narrow one must not be
// joined by the helpers it has no owner index for.
func TestPoolMoreHelpersThanWorkers(t *testing.T) {
	saved := spinWindow.Load()
	defer spinWindow.Store(saved)
	spinWindow.Store(int64(time.Second)) // keep the extra helpers awake to see the narrow jobs
	checkCover(t, 4096, 16)
	for rep := 0; rep < 100; rep++ {
		checkCover(t, 64, 2)
	}
	runtime.Gosched()
}

// TestPoolBusyFallsBackToFork: with the pool held, a call forks its own
// goroutines and still covers its range.
func TestPoolBusyFallsBackToFork(t *testing.T) {
	pool.busy.Lock()
	checkCover(t, 1000, 4)
	pool.busy.Unlock()
}

// TestPoolHelpersShareTheWork: parked helpers are woken and take blocks. Each
// block sleeps, so the caller alone would run them one at a time; the test
// asserts some ran at the same time. Without it a pool that never woke its
// helpers would pass every other test, the caller doing all the work.
func TestPoolHelpersShareTheWork(t *testing.T) {
	saved := spinWindow.Load()
	defer spinWindow.Store(saved)
	for _, win := range []time.Duration{0, time.Second} {
		spinWindow.Store(int64(win))
		checkCover(t, 64, 4) // start the helpers
		time.Sleep(time.Millisecond)
		var in, peak atomic.Int32
		parallelFor(16, 4, func(lo, hi int) {
			c := in.Add(1)
			for p := peak.Load(); c > p && !peak.CompareAndSwap(p, c); p = peak.Load() {
			}
			time.Sleep(2 * time.Millisecond)
			in.Add(-1)
		})
		if peak.Load() < 2 {
			t.Fatalf("spinWindow=%v: blocks never ran concurrently (peak %d)", win, peak.Load())
		}
	}
}
