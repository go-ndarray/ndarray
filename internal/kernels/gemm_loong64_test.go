package kernels

import "testing"

// TestScalarFallbackLoong64 runs the GEMM, dot and elementwise suites with
// LASX forced off, so the path a CPU without LASX takes is exercised on one
// that has it.
func TestScalarFallbackLoong64(t *testing.T) {
	saved := useLASX
	useLASX = false
	defer func() { useLASX = saved }()
	t.Run("gemm", TestGemmCorrectness)
	t.Run("dot", TestDotSIMD)
	t.Run("bin", TestBinSIMD)
}

func TestLASXDetected(t *testing.T) {
	t.Logf("useLASX = %v (AT_HWCAP)", useLASX)
}
