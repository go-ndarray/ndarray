package kernels

import "testing"

// TestGemm32NeedsRVV: without the V extension there is no float32 kernel, and
// HasGemm32 sends float32 products to the float64 GEMM. With it, the tile is
// 8x16 on a core whose vector registers hold sixteen floats in two (VLEN >= 256)
// and 6x16 on a narrower one.
func TestGemm32NeedsRVV(t *testing.T) {
	if newGemm32(false) != nil {
		t.Fatal("a float32 GEMM without V")
	}
	want := 6
	if vlenb >= gemm32WideVLEN {
		want = 8
	}
	t.Logf("vlenb = %d", vlenb)
	if g := newGemm32(true); g == nil || g.mr != want || g.nr != 16 {
		t.Fatalf("newGemm32(true) = %+v, want mr %d", g, want)
	}
}

// TestGemm32NarrowKernel runs the 6x16 kernel, the one a VLEN-128 core uses, on
// whatever core this is, through the float32 GEMM tests.
func TestGemm32NarrowKernel(t *testing.T) {
	if !useRVV {
		t.Skip("no V extension")
	}
	saved := gemm32
	defer func() { gemm32 = saved }()
	gemm32 = &gemmKernel[float32]{mr: 6, nr: 16, micro: gemmMicro32}
	TestGemm32Correctness(t)
	TestGemm32Accuracy(t)
	TestGemm32Specials(t)
}

// TestGemm32WideKernel runs the 8x16 kernel when the core can: VLEN >= 256.
func TestGemm32WideKernel(t *testing.T) {
	if vlenb < gemm32WideVLEN {
		t.Skip("VLEN < 256")
	}
	saved := gemm32
	defer func() { gemm32 = saved }()
	gemm32 = &gemmKernel[float32]{mr: 8, nr: 16, micro: gemmMicro32x8}
	TestGemm32Correctness(t)
	TestGemm32Accuracy(t)
	TestGemm32Specials(t)
}
