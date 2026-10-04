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

// TestGemmParallelYielding runs the parallel GEMM with every wait yielding at
// once, as on an oversubscribed machine, and with more workers than CPUs.
func TestGemmParallelYielding(t *testing.T) {
	saved := spinIters
	spinIters = 0
	defer func() { spinIters = saved }()
	withMaxProcs(8, func() {
		withThresholds(1<<14, 1, func() {
			for _, s := range []struct{ m, k, n int }{{50, 70, 90}, {13, 300, 7}, {97, 31, 600}} {
				a, b := intMat(s.m, s.k, 3), intMat(s.k, s.n, 4)
				assertGemm(t, s.m, s.k, s.n, a, b)
				withBlocks(MR, 8, 2*NR, func() { assertGemm(t, s.m, s.k, s.n, a, b) })
			}
		})
	})
}
