package kernels

import (
	"sync/atomic"
	"testing"
)

// TestSpinUntil waits on a condition another goroutine makes true, both
// spinning and yielding from the first poll.
func TestSpinUntil(t *testing.T) {
	for _, iters := range []int{spinIters, 0} {
		saved := spinIters
		spinIters = iters
		var flag atomic.Bool
		go func() {
			for i := 0; i < 1000; i++ {
				_ = i * i
			}
			flag.Store(true)
		}()
		spinUntil(flag.Load)
		spinIters = saved
	}
}
