package ndarray

import (
	"math"
	"sync"
	"testing"
)

// wsPass runs one pass of operations covering every allocation site that can
// draw from a workspace, starting from inputs bound to ws (nil = heap), and
// returns the results' data in order.
func wsPass(t *testing.T, ws *Workspace) [][]float64 {
	t.Helper()
	bind := func(a *Array) *Array {
		if ws == nil {
			return a
		}
		return ws.Use(a)
	}
	mk := func(seed float64, shape ...int) *Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		d := make([]float64, n)
		for i := range d {
			d[i] = math.Mod(float64(i)*0.37+seed, 7) - 3
		}
		return bind(mustArr(t, ok(FromData(d, shape...))))
	}
	a, b, row := mk(1, 6, 5), mk(2, 6, 5), mk(3, 1, 5)
	v, m := mk(4, 5), mk(5, 5, 4)
	var out []*Array
	add := func(x *Array, err error) *Array {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, x)
		return x
	}
	s := add(a.Add(b))             // same-shape fast path
	add(s.Mul(row))                // broadcast path (broadcastTo)
	add(a.Sub(b))                  //
	add(a.Abs().Sqrt(), nil)       // Map + Sqrt
	add(a.Transpose().Copy(), nil) // materialize, strided gather
	add(a.MatMul(m))               // GEMM, needs zeroed dst
	add(a.Inner(b))                // GEMM on a transposed view
	add(a.Dot(v))                  // MatVec
	add(v.Dot(m))                  // VecMat, needs zeroed dst
	add(v.Dot(v))                  // scalar
	add(v.Outer(v), nil)           //
	add(a.SumAxis(0, false))       // axis reduction
	add(a.MaxAxis(1, true))        //
	add(a.CumSum(1))               // scan
	add(a.CumSumFlat(), nil)       //
	add(a.CumProdFlat(), nil)      //
	add(a.Clip(-1, 1))             //
	gt := mustArr(t, ok(a.Greater(b)))
	add(Where(gt, a, b)) //
	add(Concatenate([]*Array{a, b}, 0))
	add(a.Take(0, 3, -1)) //
	add(a.Reshape(5, 6))  // view
	add(a.Ravel(), nil)   //
	add(a.Flatten(), nil) //
	add(a.MaskSelect(gt)) //
	res := make([][]float64, len(out))
	for i, x := range out {
		res[i] = x.Copy().Detach().data
	}
	return res
}

// TestWorkspaceMatchesHeap: every result computed in a workspace is the same,
// bit for bit, as on the heap — on the first pass (fresh blocks) and on a pass
// after Reset whose recycled memory was first poisoned with NaN. An operation
// that relied on zeroed memory while asking for unzeroed would leak the NaN.
func TestWorkspaceMatchesHeap(t *testing.T) {
	want := wsPass(t, nil)
	ws := NewWorkspace()
	check := func(pass string, got [][]float64) {
		t.Helper()
		for i := range want {
			if len(got[i]) != len(want[i]) {
				t.Fatalf("%s result %d: len %d, want %d", pass, i, len(got[i]), len(want[i]))
			}
			for j := range want[i] {
				if math.Float64bits(got[i][j]) != math.Float64bits(want[i][j]) {
					t.Fatalf("%s result %d [%d]: %v, want %v", pass, i, j, got[i][j], want[i][j])
				}
			}
		}
	}
	check("first pass", wsPass(t, ws))
	for pass := 0; pass < 3; pass++ {
		ws.Reset()
		for _, c := range ws.chunks {
			for i := range c {
				c[i] = math.NaN()
			}
		}
		check("poisoned pass", wsPass(t, ws))
	}
	if len(ws.chunks) != 1 {
		t.Errorf("after repeated passes the workspace holds %d blocks, want 1", len(ws.chunks))
	}
}

// TestWorkspaceSettlesIntoOneBlock: a pass that outgrows the first block makes
// several; Reset folds them into one sized to the pass, and the next identical
// pass fits in it without growing.
func TestWorkspaceSettlesIntoOneBlock(t *testing.T) {
	var ws Workspace // the zero value is usable
	pass := func() {
		for i := 0; i < 5; i++ {
			ws.alloc(minChunk/2+3, false)
		}
		ws.alloc(5*minChunk, true) // larger than double the last block
	}
	pass()
	if len(ws.chunks) < 2 {
		t.Fatalf("pass made %d blocks, want several", len(ws.chunks))
	}
	ws.Reset()
	if len(ws.chunks) != 1 {
		t.Fatalf("Reset left %d blocks, want 1", len(ws.chunks))
	}
	size := len(ws.chunks[0])
	pass()
	if len(ws.chunks) != 1 || len(ws.chunks[0]) != size {
		t.Errorf("second pass grew the workspace: %d blocks, first %d (was %d)",
			len(ws.chunks), len(ws.chunks[0]), size)
	}
	ws.Reset() // single-block pass: nothing to fold
	if len(ws.chunks) != 1 || len(ws.chunks[0]) != size {
		t.Errorf("Reset after a one-block pass changed the block")
	}
}

// TestWorkspaceAllocShape: every allocation starts 64-byte aligned within its
// block, has exactly the requested length and capacity (an append cannot
// spill into the neighbour), and zero=true clears recycled memory.
func TestWorkspaceAllocShape(t *testing.T) {
	ws := NewWorkspace()
	a := ws.alloc(3, false)
	b := ws.alloc(5, false)
	if len(a) != 3 || cap(a) != 3 || len(b) != 5 || cap(b) != 5 {
		t.Fatalf("len/cap a=%d/%d b=%d/%d", len(a), cap(a), len(b), cap(b))
	}
	if &ws.chunks[0][8] != &b[0] {
		t.Errorf("second allocation does not start at element 8 of the block")
	}
	for i := range b {
		b[i] = 7
	}
	ws.Reset()
	ws.alloc(3, false)
	if c := ws.alloc(5, true); c[0] != 0 || c[4] != 0 {
		t.Errorf("zero=true returned %v", c)
	}
	if h := (*Workspace)(nil).alloc(4, false); len(h) != 4 {
		t.Errorf("nil workspace alloc len %d", len(h))
	}
}

// TestWorkspaceDetachAndUse: Detach is a heap copy that survives Reset; Use
// leaves its argument unbound and shares its data; views stay bound.
func TestWorkspaceDetachAndUse(t *testing.T) {
	ws := NewWorkspace()
	x := mustArr(t, ok(FromData([]float64{1, 2, 3, 4}, 2, 2)))
	bx := ws.Use(x)
	if x.ws != nil || bx.ws != ws || &bx.data[0] != &x.data[0] {
		t.Fatal("Use must bind a view sharing x's data and leave x unbound")
	}
	if bx.Transpose().ws != ws {
		t.Error("a view of a bound array is not bound")
	}
	y := mustArr(t, ok(bx.Add(bx)))
	kept := y.Detach()
	if kept.ws != nil {
		t.Error("Detach returned a bound array")
	}
	ws.Reset()
	for i := range ws.chunks[0] {
		ws.chunks[0][i] = -1
	}
	wantData(t, kept, []int{2, 2}, []float64{2, 4, 6, 8})
	wantData(t, x, []int{2, 2}, []float64{1, 2, 3, 4})
}

// TestWorkspaceConcurrent: allocation from several goroutines is serialized,
// so concurrent operations on arrays bound to one workspace never overlap.
func TestWorkspaceConcurrent(t *testing.T) {
	ws := NewWorkspace()
	x := ws.Use(mustArr(t, ok(Arange(0, 1000, 1))))
	var wg sync.WaitGroup
	results := make([]*Array, 8)
	for g := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := x.Add(x)
			if err != nil {
				t.Error(err)
				return
			}
			for i := range r.data {
				r.data[i] += float64(g)
			}
			results[g] = r
		}()
	}
	wg.Wait()
	for g, r := range results {
		for i, v := range r.data {
			if v != float64(2*i+g) {
				t.Fatalf("goroutine %d [%d] = %v: results overlap", g, i, v)
			}
		}
	}
}
