package blasmm

import "testing"

// view lays a rows x cols matrix of small integers out with the given strides
// in a buffer padded with a sentinel, from offset off; the products below are
// exact, so every comparison is equality.
func view[T float32 | float64](rows, cols, off, rs, cs int, seed int) (Mat[T], [][]T) {
	hi := off // highest element index
	for _, ext := range [][2]int{{rows - 1, rs}, {cols - 1, cs}} {
		if ext[1] > 0 {
			hi += ext[0] * ext[1]
		}
	}
	buf := make([]T, hi+1)
	for i := range buf {
		buf[i] = 9999 // never an element, or a product would be off
	}
	want := make([][]T, rows)
	for i := range want {
		want[i] = make([]T, cols)
		for j := range want[i] {
			v := T((i*7+j*3+seed)%5 - 2)
			want[i][j] = v
			buf[off+i*rs+j*cs] = v // an aliased view (rs = 0) keeps the last write
		}
	}
	for i := range want { // re-read: aliasing makes the matrix what the buffer says
		for j := range want[i] {
			want[i][j] = buf[off+i*rs+j*cs]
		}
	}
	return Mat[T]{Data: buf, Off: off, RS: rs, CS: cs}, want
}

func product[T float32 | float64](a, b [][]T, m, k, n int) []T {
	c := make([]T, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			for p := 0; p < k; p++ {
				c[i*n+j] += a[i][p] * b[p][j]
			}
		}
	}
	return c
}

func testMatMul[T float32 | float64](t *testing.T, mul func(dst []T, a, b Mat[T], m, k, n int)) {
	type lay struct {
		name        string
		off, rs, cs func(rows, cols int) int
	}
	layouts := []lay{
		{"row-major", func(r, c int) int { return 0 }, func(r, c int) int { return c }, func(r, c int) int { return 1 }},
		{"padded", func(r, c int) int { return 5 }, func(r, c int) int { return c + 3 }, func(r, c int) int { return 1 }},
		{"transposed", func(r, c int) int { return 0 }, func(r, c int) int { return 1 }, func(r, c int) int { return r }},
		{"every other column", func(r, c int) int { return 0 }, func(r, c int) int { return 2 * c }, func(r, c int) int { return 2 }},
		{"every other row", func(r, c int) int { return 0 }, func(r, c int) int { return 2 * c }, func(r, c int) int { return 1 }},
		{"reversed rows", func(r, c int) int { return (r - 1) * c }, func(r, c int) int { return -c }, func(r, c int) int { return 1 }},
		{"reversed columns", func(r, c int) int { return c - 1 }, func(r, c int) int { return c }, func(r, c int) int { return -1 }},
		{"broadcast rows", func(r, c int) int { return 0 }, func(r, c int) int { return 0 }, func(r, c int) int { return 1 }},
	}
	shapes := [][3]int{{1, 1, 1}, {1, 5, 4}, {4, 5, 1}, {3, 1, 6}, {7, 5, 3}, {13, 17, 19}, {40, 33, 50}}
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		for _, la := range layouts {
			for _, lb := range layouts {
				a, aw := view[T](m, k, la.off(m, k), la.rs(m, k), la.cs(m, k), 1)
				b, bw := view[T](k, n, lb.off(k, n), lb.rs(k, n), lb.cs(k, n), 2)
				got := make([]T, m*n)
				for i := range got {
					got[i] = 7777 // overwritten, not accumulated
				}
				mul(got, a, b, m, k, n)
				want := product(aw, bw, m, k, n)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%v a=%s b=%s: [%d] = %v, want %v", sh, la.name, lb.name, i, got[i], want[i])
					}
				}
			}
		}
	}
	// A stride of an axis of extent 1 is never used, whatever it is.
	a := Mat[T]{Data: []T{2, 3}, RS: 0, CS: 1}  // 1 x 2, rs unused
	b := Mat[T]{Data: []T{4, 5}, RS: 1, CS: -3} // 2 x 1, cs unused
	got := make([]T, 1)
	mul(got, a, b, 1, 2, 1)
	if got[0] != 23 {
		t.Fatalf("extent-1 strides: %v, want 23", got[0])
	}
	// Empty products write nothing; k = 0 is the zero matrix.
	mul(nil, Mat[T]{}, Mat[T]{}, 0, 3, 4)
	mul(nil, Mat[T]{}, Mat[T]{}, 3, 3, 0)
	z := []T{5, 5, 5, 5}
	mul(z, Mat[T]{}, Mat[T]{}, 2, 0, 2)
	for _, v := range z {
		if v != 0 {
			t.Fatalf("k = 0: %v, want zeros", z)
		}
	}
}

func TestMatMul(t *testing.T) { testMatMul(t, MatMul) }

func TestMatMul32(t *testing.T) {
	t.Logf("HasGemm32() = %v; the float32 product is tested where there is a kernel", HasGemm32())
	if HasGemm32() {
		testMatMul(t, MatMul32)
	}
}

// A refusal by go-blas is a defect of this package and panics with its name.
func TestMatMulPanicsOnADefect(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a destination too short was accepted")
		}
	}()
	one := Mat[float64]{Data: []float64{1}, RS: 1, CS: 1}
	MatMul(make([]float64, 2), one, one, 1, 1, 2) // B is 1 x 2 but holds one element
}
