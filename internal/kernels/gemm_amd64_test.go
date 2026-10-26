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

// TestGemm32NeedsFMA: without FMA there is no float32 kernel, and HasGemm32
// sends float32 products to the float64 GEMM.
func TestGemm32NeedsFMA(t *testing.T) {
	if newGemm32(false) != nil {
		t.Fatal("a float32 GEMM without FMA")
	}
	if g := newGemm32(true); g == nil || g.mr != 6 || g.nr != 16 {
		t.Fatalf("newGemm32(true) = %+v", g)
	}
}
