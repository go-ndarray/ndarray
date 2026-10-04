package kernels

import "testing"

// TestGemmSSE2Fallback runs the GEMM suite on the baseline SSE2 kernel whatever
// the host supports, so a CPU with FMA still exercises the path a pre-2013 CPU
// (or a hypervisor that masks FMA) takes. The FMA path is the default and is
// exercised by the rest of the suite wherever hasFMA is true.
func TestGemmSSE2Fallback(t *testing.T) {
	saved := useFMA
	useFMA = false
	defer func() { useFMA = saved }()
	t.Run("correctness", TestGemmCorrectness)
	t.Run("bands", TestGemmBandSizing)
	t.Run("float", TestGemmFloatTolerance)
	t.Run("dot", TestDotSIMD)
}
