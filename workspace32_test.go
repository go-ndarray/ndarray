package ndarray

import (
	"math"
	"testing"
	"unsafe"
)

// wsPass32 is wsPass for the Float32 allocation sites (unary32, mathUnary,
// arith, power, clip, the float32 GEMM), returning the results' bits.
func wsPass32(t *testing.T, ws *Workspace) [][]uint32 {
	t.Helper()
	mk := func(seed float64, shape ...int) *Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		d := make([]float64, n)
		for i := range d {
			d[i] = math.Mod(float64(i)*0.37+seed, 7) - 3
		}
		a := mustArr(t, ok(FromData(d, shape...))).AsType(Float32)
		if ws != nil {
			a = ws.Use(a)
		}
		return a
	}
	a, b, row := mk(1, 6, 5), mk(2, 6, 5), mk(3, 1, 5)
	m := mk(5, 5, 4)
	var out []*Array
	add := func(x *Array, err error) *Array {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if x.DType() != Float32 {
			t.Fatalf("result %d is %v, want float32", len(out), x.DType())
		}
		out = append(out, x)
		return x
	}
	s := add(a.Add(b))       // arith, same shape
	add(s.Mul(row))          // arith, broadcast
	add(a.Abs().Exp(), nil)  // unary32
	add(a.Abs().Log(), nil)  // unary32
	add(a.Sin(), nil)        // mathUnary
	add(a.Abs().Sqrt(), nil) // mathUnary (Sqrt32)
	add(a.Power(3), nil)     // power
	add(a.Clip(-1, 1))       // clipTyped
	add(a.MatMul(m))         // float32 GEMM, needs zeroed dst
	add(s.MatMul(m))         // a second GEMM on recycled memory
	add(a.Abs().Log().Exp(), nil)
	res := make([][]uint32, len(out))
	for i, x := range out {
		d := x.Copy().Detach().contiguousStore().([]float32)
		res[i] = make([]uint32, len(d))
		for j, v := range d {
			res[i][j] = math.Float32bits(v)
		}
	}
	return res
}

// TestWorkspace32MatchesHeap: every Float32 result computed in a workspace is
// the heap's, bit for bit, on the first pass and on passes whose recycled
// memory was first filled with all-ones bits (NaN in both float32 halves of a
// float64): a site that relied on zeroed memory would leak them.
func TestWorkspace32MatchesHeap(t *testing.T) {
	want := wsPass32(t, nil)
	ws := NewWorkspace()
	check := func(pass string, got [][]uint32) {
		t.Helper()
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("%s result %d: len %d, want %d", pass, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("%s result %d [%d]: 0x%08x, want 0x%08x", pass, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
	check("first pass", wsPass32(t, ws))
	for range 3 {
		ws.Reset()
		for _, c := range ws.chunks {
			for i := range c {
				c[i] = math.Float64frombits(^uint64(0))
			}
		}
		check("poisoned pass", wsPass32(t, ws))
	}
}

// TestWorkspaceAlloc32Shape: a Float32 allocation has exactly the requested
// length and capacity, odd lengths included, starts on a 64-byte boundary of
// its block, and zero=true clears recycled memory.
func TestWorkspaceAlloc32Shape(t *testing.T) {
	ws := NewWorkspace()
	a := ws.alloc32(3, false)
	b := ws.alloc32(17, false)
	if len(a) != 3 || cap(a) != 3 || len(b) != 17 || cap(b) != 17 {
		t.Fatalf("len/cap a=%d/%d b=%d/%d", len(a), cap(a), len(b), cap(b))
	}
	if got, want := &b[0], (*float32)(unsafe.Pointer(&ws.chunks[0][8])); got != want {
		t.Errorf("second allocation does not start at element 8 of the block")
	}
	for i := range b {
		b[i] = 7
	}
	ws.Reset()
	ws.alloc32(3, false)
	if c := ws.alloc32(17, true); c[0] != 0 || c[16] != 0 {
		t.Errorf("zero=true returned %v", c)
	}
	if h := (*Workspace)(nil).alloc32(4, false); len(h) != 4 {
		t.Errorf("nil workspace alloc32 len %d", len(h))
	}
	if z := ws.alloc32(0, false); len(z) != 0 {
		t.Errorf("alloc32(0) len %d", len(z))
	}
}

// TestWorkspace32Allocates: after a warm-up pass, a Float32 Log in a
// workspace allocates less than on the heap (the result's storage comes from
// the arena; only the Array header and shape remain).
func TestWorkspace32Allocates(t *testing.T) {
	x := mustArr(t, ok(FromData(make([]float64, 4096), 4096))).AsType(Float32)
	heap := testing.AllocsPerRun(20, func() { _ = x.Log() })
	ws := NewWorkspace()
	xw := ws.Use(x)
	_ = xw.Log()
	ws.Reset()
	inWS := testing.AllocsPerRun(20, func() {
		_ = xw.Log()
		ws.Reset()
	})
	if inWS >= heap {
		t.Errorf("Float32 Log allocates %v times per call in a workspace, %v on the heap", inWS, heap)
	}
}
