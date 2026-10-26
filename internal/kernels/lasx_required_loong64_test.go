package kernels

import (
	"os"
	"testing"
)

// TestLASXWhenRequired fails when NDARRAY_REQUIRE_LASX is set and the CPU does
// not report LASX. CI sets it on the loong64 qemu lane, so the LSX and LASX
// kernels cannot go untested behind a green tick there, the way the RVV ones
// did on riscv64 until 2026-10-26.
func TestLASXWhenRequired(t *testing.T) {
	t.Logf("useLASX = %v", useLASX)
	if os.Getenv("NDARRAY_REQUIRE_LASX") != "" && !useLASX {
		t.Fatal("NDARRAY_REQUIRE_LASX is set but the CPU does not report LASX: the LSX/LASX kernels did not run")
	}
}
