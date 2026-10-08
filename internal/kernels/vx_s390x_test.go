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

	})
}
