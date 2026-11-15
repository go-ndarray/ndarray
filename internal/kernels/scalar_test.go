package kernels

import (
	"math"
	"testing"
)

// TestScalarPSameBits: streaming the scalar block by block gives, bit for
// bit, what the kernel gives on the scalar materialised to full length, for
// both operand orders, at every block boundary and across ParThreshold.
func TestScalarPSameBits(t *testing.T) {
	ks := map[string]func(dst, a, b []float64){"Add": AddP, "Sub": SubP, "Mul": MulP, "Div": DivP}
	const s = 1.0 / 3
	for _, n := range []int{0, 1, 7, scalarBlock - 1, scalarBlock, scalarBlock + 1, 3*scalarBlock + 5, ParThreshold + 3} {
		full := make([]float64, n)
		for i := range full {
			full[i] = math.Sin(float64(i)) * 7
		}
		wide := make([]float64, n)
		for i := range wide {
			wide[i] = s
		}
		for name, k := range ks {
			for _, sFirst := range []bool{false, true} {
				want := make([]float64, n)
				if sFirst {
					k(want, wide, full)
				} else {
					k(want, full, wide)
				}
				got := make([]float64, n)
				ScalarP(k, got, full, s, sFirst)
				// In place, as AddInto(out = x) does: dst is full itself.
				inPlace := append([]float64(nil), full...)
				ScalarP(k, inPlace, inPlace, s, sFirst)
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("%s n=%d sFirst=%v [%d]: got %v, want %v", name, n, sFirst, i, got[i], want[i])
					}
					if math.Float64bits(inPlace[i]) != math.Float64bits(want[i]) {
						t.Fatalf("%s n=%d sFirst=%v in place [%d]: got %v, want %v", name, n, sFirst, i, inPlace[i], want[i])
					}
				}
			}
		}
	}
}

// TestFill: every element is set, for lengths around the powers of two the
// doubling copy steps through, and nothing past len(dst) is touched.
func TestFill(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 63, 64, 65, 1000} {
		buf := make([]float64, n+1)
		buf[n] = -1
		Fill(buf[:n], 2.5)
		for i := range n {
			if buf[i] != 2.5 {
				t.Fatalf("n=%d: [%d] = %v", n, i, buf[i])
			}
		}
		if buf[n] != -1 {
			t.Fatalf("n=%d: wrote past the end", n)
		}
	}
}
