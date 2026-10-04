package kernels

import (
	"sync"
	"sync/atomic"
	"testing"
)

// TestSpinBarrier: no goroutine leaves a generation before all have arrived,
// over many back-to-back generations, both spinning and yielding.
func TestSpinBarrier(t *testing.T) {
	for _, iters := range []int{spinIters, 0} {
		saved := spinIters
		spinIters = iters
		const n, rounds = 4, 2000
		bar := newSpinBarrier(n)
		var arrived atomic.Int64
		var bad atomic.Int64
		var wg sync.WaitGroup
		for g := 0; g < n; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for r := int64(1); r <= rounds; r++ {
					arrived.Add(1)
					bar.wait()
					// Everyone has arrived for round r, and nobody can have
					// arrived for round r+1 before this goroutine has.
					if got := arrived.Load(); got < r*n || got > r*n+n-1 {
						bad.Add(1)
					}
					bar.wait()
				}
			}()
		}
		wg.Wait()
		spinIters = saved
		if bad.Load() != 0 {
			t.Fatalf("spinIters=%d: %d early departures", iters, bad.Load())
		}
	}
}
