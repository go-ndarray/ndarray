package kernels

import "testing"

// TestVXDetected logs whether this machine runs the VX kernels, and runs the
// SIMD-vs-scalar checks again with them forced off.
func TestVXDetected(t *testing.T) {
	t.Logf("useVX = %v (AT_HWCAP)", useVX)
	saved := useVX
	useVX = false
	defer func() { useVX = saved }()
	t.Run("scalar", func(t *testing.T) {
		TestBinSIMD(t)
		TestSumSIMD(t)
		TestGemmParallelYielding(t) // gemmMicroGo against the oracle

	})
}

// TestVXEDetected runs the max/min checks with the vector kernel forced off
// too, and logs which path this machine takes.
func TestVXEDetected(t *testing.T) {
	t.Logf("useVXE = %v (AT_HWCAP)", useVXE)
	saved := useVXE
	useVXE = false
	defer func() { useVXE = saved }()
	t.Run("scalar", func(t *testing.T) {
		TestMaxMinSIMD(t)
		TestMaxMinSIMDSpecialsInVectorLanes(t)
	})
}
