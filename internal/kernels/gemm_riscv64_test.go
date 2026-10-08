package kernels

import "testing"

// TestRVVDetected logs whether this machine runs the RVV kernels, and runs the
// SIMD-vs-scalar checks again with them forced off.
func TestRVVDetected(t *testing.T) {
	t.Logf("useRVV = %v (AT_HWCAP)", useRVV)
	saved := useRVV
	useRVV = false
	defer func() { useRVV = saved }()
	t.Run("scalar", func(t *testing.T) {
		TestBinSIMD(t)
		TestSumSIMD(t)
		TestGemmParallelYielding(t) // gemmMicroGo against the oracle
	})
}
