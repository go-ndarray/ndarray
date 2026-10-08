//go:build amd64 || arm64 || ppc64le || loong64 || riscv64 || s390x

package kernels

import "testing"

// The wrappers check the operands the assembly reads against the length it is
// given, so a caller's mistake is an index panic in Go rather than an access
// past a slice inside the kernel.
func TestWrappersRefuseShortOperands(t *testing.T) {
	long, short := make([]float64, 8), make([]float64, 7)
	ldc := NR
	for name, f := range map[string]func(){
		"sqrt src": func() { sqrtSIMD(long, short) },
		"add a":    func() { addBin(long, short, long) },
		"sub b":    func() { subBin(long, long, short) },
		"mul a":    func() { mulBin(long, short, long) },
		"div b":    func() { divBin(long, long, short) },
		"dot b":    func() { dotSIMD(long, short) },
		"gemm pa":  func() { gemmMicro(1, make([]float64, MR-1), make([]float64, NR), make([]float64, (MR-1)*ldc+NR), ldc) },
		"gemm pb":  func() { gemmMicro(1, make([]float64, MR), make([]float64, NR-1), make([]float64, (MR-1)*ldc+NR), ldc) },
		"gemm dst": func() { gemmMicro(1, make([]float64, MR), make([]float64, NR), make([]float64, (MR-1)*ldc+NR-1), ldc) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: a short operand was accepted", name)
				}
			}()
			f()
		}()
	}
}
