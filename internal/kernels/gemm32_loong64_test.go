package kernels

import "testing"

// TestGemm32NeedsLASX: without LASX there is no float32 kernel, and HasGemm32
// sends float32 products to the float64 GEMM.
func TestGemm32NeedsLASX(t *testing.T) {
	if newGemm32(false) != nil {
		t.Fatal("a float32 GEMM without LASX")
	}
	if g := newGemm32(true); g == nil || g.mr != 8 || g.nr != 16 {
		t.Fatalf("newGemm32(true) = %+v", g)
	}
}
