package ndarray

import "sync"

// Workspace is an arena for the results of array operations, for loops that
// compute the same shapes over and over. Bind an input with Use; every result
// computed from a bound array (and every result computed from those) takes its
// memory from the workspace instead of the Go heap, and Reset hands all of it
// back at once:
//
//	ws := ndarray.NewWorkspace()
//	for step := 0; step < n; step++ {
//		x := ws.Use(state)
//		y, _ := x.Mul(w)       // from the workspace
//		z, _ := y.Add(bias)    // from the workspace
//		state = z.Detach()     // the one result that must outlive the step
//		ws.Reset()
//	}
//
// After the first pass the arena holds the peak footprint of one pass in a
// single block, and later passes allocate nothing: no garbage-collector
// cycles, no zeroing of memory the kernel overwrites anyway, and the same
// cache-warm bytes every time. This is the workspace pattern of ND4J, the
// Java ndarray library, and the scoped counterpart of NumPy's reference-count
// free-list, which a garbage-collected language does not have. The *Into
// methods (AddInto, SqrtInto, ...) remain the way to reuse one named buffer.
//
// ⚠ Reset invalidates every array allocated from the workspace since the
// previous Reset, and every view of one: their memory is reused by the next
// results, so reading them afterwards returns other values. Detach (a heap
// copy, unbound) is how a result leaves the workspace. Arrays bound with Use
// but not allocated from it (the inputs) are not affected.
//
// A Workspace may be used from several goroutines; allocation is serialized.
// The zero value is an empty workspace ready to use.
type Workspace struct {
	mu     sync.Mutex
	chunks [][]float64 // backing blocks; chunks[cur], the last, is being carved
	cur    int         // index of the block being carved
	off    int         // elements of chunks[cur] already handed out
	used   int         // elements handed out since the last Reset, all blocks
}

// minChunk is the smallest block the arena asks the heap for (1 MiB), so a
// pass of small results does not grow it one result at a time.
const minChunk = 1 << 17

// NewWorkspace returns an empty workspace. It holds no memory until the first
// allocation.
func NewWorkspace() *Workspace { return &Workspace{} }

// Use returns a view of a (sharing its data) that is bound to w: the results
// of operations on it are allocated from w. a itself is not modified, and its
// data stays where it is.
func (w *Workspace) Use(a *Array) *Array {
	v := *a
	v.ws = w
	return &v
}

// Reset makes all memory allocated from w since the last Reset available
// again (see the warning on Workspace). If the pass needed more than one
// block, they are replaced by a single block of the pass's total, so a
// repeated pass settles into one contiguous block.
func (w *Workspace) Reset() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cur > 0 {
		w.chunks = [][]float64{make([]float64, w.used)}
	}
	w.cur, w.off, w.used = 0, 0, 0
}

// alloc returns n elements from the workspace. Each allocation starts on a
// 64-byte boundary of its block (8 elements), so a result never shares a
// cache line with its neighbour, and has cap n, so an append can never write
// into the next one. Unless zero is set, the elements hold whatever the
// previous pass left there; callers pass zero=false only when the kernel
// writes every element.
func (w *Workspace) alloc(n int, zero bool) []float64 {
	if w == nil {
		return make([]float64, n)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	need := (n + 7) &^ 7
	if len(w.chunks) == 0 || w.off+need > len(w.chunks[w.cur]) {
		// Start a new block, at least double the last: the blocks only ever
		// grow within a pass, and Reset folds them back into one.
		size := max(need, minChunk)
		if len(w.chunks) > 0 {
			size = max(size, 2*len(w.chunks[w.cur]))
		}
		w.chunks = append(w.chunks, make([]float64, size))
		w.cur, w.off = len(w.chunks)-1, 0
	}
	s := w.chunks[w.cur][w.off : w.off+n : w.off+n]
	w.off += need
	w.used += need
	if zero {
		clear(s)
	}
	return s
}

// alloc returns n elements for a result computed from a: from a's workspace
// when it is bound to one, else from the heap (always zeroed there).
func (a *Array) alloc(n int, zero bool) []float64 { return a.ws.alloc(n, zero) }

// Detach returns a heap copy of a that is not bound to any workspace: the way
// a result computed in a workspace outlives its Reset.
func (a *Array) Detach() *Array {
	u := *a
	u.ws = nil
	return u.Copy()
}

// wsOf returns the workspace of the first bound array among xs, or nil: the
// rule for functions of several arrays (Where, Concatenate, ...).
func wsOf(xs ...*Array) *Workspace {
	for _, x := range xs {
		if x.ws != nil {
			return x.ws
		}
	}
	return nil
}
