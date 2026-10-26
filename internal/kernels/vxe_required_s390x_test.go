package kernels

import (
	"os"
	"testing"
)

// TestVXEWhenRequired fails when NDARRAY_REQUIRE_VXE is set and the CPU does
// not report the vector facility and its enhancements (z14). CI sets it on the
// s390x qemu lane, so the VX and VXE kernels (the float32 ones need z14)
// cannot go untested behind a green tick there.
func TestVXEWhenRequired(t *testing.T) {
	t.Logf("useVX = %v, useVXE = %v", useVX, useVXE)
	if os.Getenv("NDARRAY_REQUIRE_VXE") != "" && !(useVX && useVXE) {
		t.Fatal("NDARRAY_REQUIRE_VXE is set but the CPU does not report VX and VXE: the vector kernels did not all run")
	}
}
