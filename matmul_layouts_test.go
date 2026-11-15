package ndarray

import (
	"fmt"
	"hash/fnv"
	"math"
	"runtime"
	"testing"
)

// MatMul reads each operand through its strides (go-blas/blas), whatever the
// view: this test multiplies every pair of eleven layouts (contiguous,
// transposed, reversed on either or both axes, broadcast on either or both,
// decimated with an offset, extent-1 axes with garbage strides), for float32
// and float64 and for shapes with empty, unit and ragged extents, and checks
//
//   - every element against a naive triple loop, within a bound;
//   - that the result's bits are the same from the heap and from a Workspace
//     pre-filled with NaN (nothing reads the destination), and at GOMAXPROCS 1
//     and N (the result does not depend on the worker count);
//   - that sixteen goroutines multiplying the same views at once agree.
//
// Under -short (the emulated CI lanes) the two largest shapes are left out.

type matLayout struct {
	name string
	mk   func(rows, cols int) (n, rs, cs, off int)
}

var rvLayouts = []matLayout{
	{"contig", func(r, c int) (int, int, int, int) { return r * c, c, 1, 0 }},
	{"trans", func(r, c int) (int, int, int, int) { return r * c, 1, r, 0 }},
	{"negrow", func(r, c int) (int, int, int, int) { return r * c, -c, 1, (r - 1) * c }},
	{"negcol", func(r, c int) (int, int, int, int) { return r * c, c, -1, c - 1 }},
	{"negboth", func(r, c int) (int, int, int, int) { return r * c, -c, -1, r*c - 1 }},
	{"transneg", func(r, c int) (int, int, int, int) { return r * c, -1, r, r - 1 }},
	{"bcastrow", func(r, c int) (int, int, int, int) { return c, 0, 1, 0 }},
	{"bcastcol", func(r, c int) (int, int, int, int) { return r, 1, 0, 0 }},
	{"bcastall", func(r, c int) (int, int, int, int) { return 1, 0, 0, 0 }},
	{"step2off", func(r, c int) (int, int, int, int) {
		rs := 2*c + 3
		return 5 + (r-1)*rs + (c-1)*2 + 1, rs, 2, 5
	}},
	{"odd1", func(r, c int) (int, int, int, int) { return r * c, c, 1, 0 }},
}

func rvMake(l matLayout, rows, cols int, f32 bool, seed int) *Array {
	n, rs, cs, off := l.mk(rows, cols)
	if rows == 0 || cols == 0 {
		n, rs, cs, off = 0, max(cols, 1), 1, 0
	}
	if l.name == "odd1" {
		// extent-1 axes carry garbage strides
		if rows == 1 {
			rs = -7
		}
		if cols == 1 {
			cs = 0
		}
	}
	a := &Array{shape: []int{rows, cols}, strides: []int{rs, cs}, offset: off}
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = float64((i*37+seed*11)%101-50) / 7
	}
	if f32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(vals[i])
		}
		a.ext, a.dtype = d, Float32
	} else {
		a.data, a.dtype = vals, Float64
	}
	return a
}

func rvAt(a *Array, i, j int) float64 {
	p := a.offset + i*a.strides[0] + j*a.strides[1]
	if a.dtype == Float32 {
		return float64(a.ext.([]float32)[p])
	}
	return a.data[p]
}

func rvHash(r *Array) uint64 {
	h := fnv.New64a()
	m, n := r.shape[0], r.shape[1]
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var b [8]byte
			u := math.Float64bits(rvAt(r, i, j))
			for q := 0; q < 8; q++ {
				b[q] = byte(u >> (8 * q))
			}
			h.Write(b[:])
		}
	}
	return h.Sum64()
}

func rvRun(t *testing.T, ws *Workspace) []string {
	shapes := [][3]int{{0, 3, 4}, {3, 0, 4}, {3, 4, 0}, {1, 1, 1}, {1, 5, 1}, {5, 1, 7}, {1, 9, 6}, {6, 9, 1},
		{3, 4, 5}, {17, 13, 11}, {33, 31, 29}, {64, 64, 64}, {70, 1, 90}}
	if !testing.Short() {
		shapes = append(shapes, [3]int{1, 300, 200}, [3]int{130, 67, 150}, [3]int{257, 129, 65})
	}
	var out []string
	for _, f32 := range []bool{false, true} {
		for _, s := range shapes {
			m, k, n := s[0], s[1], s[2]
			for _, la := range rvLayouts {
				for _, lb := range rvLayouts {
					a := rvMake(la, m, k, f32, 1)
					b := rvMake(lb, k, n, f32, 2)
					if ws != nil {
						a = ws.Use(a)
					}
					r, err := a.MatMul(b)
					if err != nil {
						t.Fatalf("%v %s %s: %v", s, la.name, lb.name, err)
					}
					// reference
					var worst float64
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							var ref, mag float64
							for p := 0; p < k; p++ {
								ref += rvAt(a, i, p) * rvAt(b, p, j)
								mag += math.Abs(rvAt(a, i, p) * rvAt(b, p, j))
							}
							got := rvAt(r, i, j)
							tol := 1e-12 * (mag + 1)
							if f32 {
								tol = 1e-5 * (mag + 1)
							}
							if d := math.Abs(got - ref); !(d <= tol) {
								worst = math.Max(worst, d)
								if math.IsNaN(d) {
									worst = math.Inf(1)
								}
							}
						}
					}
					key := fmt.Sprintf("f32=%v %dx%dx%d %s*%s", f32, m, k, n, la.name, lb.name)
					if worst != 0 {
						t.Errorf("%s: off by %g", key, worst)
					}
					out = append(out, fmt.Sprintf("%s %016x", key, rvHash(r)))
					if ws != nil {
						ws.Reset()
					}
				}
			}
		}
	}
	return out
}

func rvCompare(t *testing.T, label string, x, y []string) {
	if len(x) != len(y) {
		t.Fatalf("%s: %d vs %d lines", label, len(x), len(y))
	}
	bad := 0
	for i := range x {
		if x[i] != y[i] {
			bad++
			if bad <= 10 {
				t.Errorf("%s: %q vs %q", label, x[i], y[i])
			}
		}
	}
	t.Logf("%s: %d lines, %d differ", label, len(x), bad)
}

func TestMatMulConcurrentViews(t *testing.T) {
	a := rvMake(rvLayouts[1], 130, 67, false, 1)
	b := rvMake(rvLayouts[3], 67, 150, false, 2)
	a32 := rvMake(rvLayouts[9], 130, 67, true, 1)
	b32 := rvMake(rvLayouts[0], 67, 150, true, 2)
	r0, _ := a.MatMul(b)
	s0, _ := a32.MatMul(b32)
	want, want32 := rvHash(r0), rvHash(s0)
	ws := NewWorkspace()
	done := make(chan string, 16)
	for g := 0; g < 16; g++ {
		go func(g int) {
			msg := ""
			for it := 0; it < 20; it++ {
				x := a
				if g%2 == 0 {
					x = ws.Use(a)
				}
				r, _ := x.MatMul(b)
				s, _ := a32.MatMul(b32)
				if rvHash(r) != want || rvHash(s) != want32 {
					msg = fmt.Sprintf("goroutine %d iter %d differs", g, it)
				}
			}
			done <- msg
		}(g)
	}
	for g := 0; g < 16; g++ {
		if m := <-done; m != "" {
			t.Error(m)
		}
	}
}

func TestMatMulLayouts(t *testing.T) {
	heap := rvRun(t, nil)
	// A workspace whose memory holds NaN: a product that read its destination
	// would show it.
	ws := NewWorkspace()
	junk := ws.alloc(1<<20, false)
	for i := range junk {
		junk[i] = math.NaN()
	}
	ws.Reset()
	wsr := rvRun(t, ws)
	rvCompare(t, "heap vs dirty workspace", heap, wsr)
	old := runtime.GOMAXPROCS(1)
	one := rvRun(t, nil)
	runtime.GOMAXPROCS(old)
	rvCompare(t, "GOMAXPROCS N vs 1", heap, one)
}
