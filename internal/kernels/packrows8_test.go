package kernels

import "testing"

// TestPackRows8 checks the 8-row fast path of packA against the layout it
// documents, pa[ir*kc + p*8 + r] = A(ic+ir+r, pc+p), for row strides larger
// than the packed width, an offset, and an edge panel (mr < 8) that takes the
// general path, in both element types.
func TestPackRows8(t *testing.T) {
	testPackRows8[float32](t)
	testPackRows8[float64](t)
}

func testPackRows8[T gemmFloat](t *testing.T) {
	g := &gemmKernel[T]{mr: 8, nr: 8}
	const m, k, rs, off = 21, 13, 17, 3
	data := make([]T, off+m*rs)
	for i := range data {
		data[i] = T(i + 1)
	}
	a := MatOf[T]{Data: data, Off: off, RS: rs, CS: 1}
	for _, c := range []struct{ ic, mc, pc, kc int }{
		{0, 16, 0, k}, {0, 21, 2, 9}, {5, 16, 1, 1}, {8, 13, 0, 12},
	} {
		pa := make([]T, roundUp(c.mc, 8)*c.kc)
		g.packA(a, c.ic, c.mc, c.pc, c.kc, pa)
		for ir := 0; ir < roundUp(c.mc, 8); ir += 8 {
			for p := 0; p < c.kc; p++ {
				for r := 0; r < 8; r++ {
					var want T
					if ir+r < c.mc {
						want = a.Data[off+(c.ic+ir+r)*rs+c.pc+p]
					}
					if got := pa[ir*c.kc+p*8+r]; got != want {
						t.Fatalf("%T %+v: pa[ir=%d p=%d r=%d] = %v, want %v", want, c, ir, p, r, got, want)
					}
				}
			}
		}
	}
}
