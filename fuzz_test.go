package ndarray

import (
	"math"
	"testing"
)

// FuzzOps is a differential fuzzer over the public API. Arbitrary bytes build
// arrays of small (possibly empty, possibly invalid) shapes and choose a
// sequence of operations and their arguments: slices with any bounds and
// steps, reshapes with -1, transposes, squeezes, broadcasting arithmetic, axis
// reductions, indexing, concatenation and matrix products.
//
// Every result is compared with an oracle written here from NumPy's
// definitions, independently of the library: the oracle reads the operands
// element by element with At and computes what each output element must be.
// A self-consistency check alone is blind: a view with wrong strides reads
// back the same wrong values through At and Copy. Values are small integers,
// so every sum and product is exact and the comparison is equality.
//
// The oracle also says whether the call is valid; the library must agree,
// returning an error exactly when the oracle rejects the arguments, and must
// never panic. The seed corpus runs as an ordinary test; `go test -fuzz
// FuzzOps` explores.
func FuzzOps(f *testing.F) {
	for _, s := range [][]byte{
		{},
		{0, 2, 3, 1, 1, 2, 0, 0, 3, 5},
		{1, 0, 3, 7, 0, 255, 2, 1, 9, 9, 9},
		{2, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{3, 1, 1, 3, 0, 2, 254, 1, 4, 0, 6, 2},
		{0, 3, 3, 2, 11, 1, 12, 0, 13, 2, 14},
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		r := &byteReader{b: data}
		var pool []*Array
		for range 2 {
			if a := r.array(t); a != nil {
				pool = append(pool, a)
			}
		}
		for step := 0; step < 12 && r.more() && len(pool) > 0; step++ {
			a := pool[r.n(len(pool))]
			b := pool[r.n(len(pool))]
			if res := applyChecked(t, r, a, b); res != nil && len(pool) < 6 {
				pool = append(pool, res)
			}
		}
	})
}

// applyChecked runs one operation chosen by r and checks it against the
// oracle, returning the result when there is one.
func applyChecked(t *testing.T, r *byteReader, a, b *Array) *Array {
	t.Helper()
	var (
		got  *Array
		err  error
		want *ref // nil: the oracle rejects the call
		name string
	)
	switch r.n(14) {
	case 0:
		specs := make([]sliceSpec, r.n(4))
		idx := make([]Index, len(specs))
		for i := range specs {
			specs[i] = sliceSpec{kind: r.n(4), x: r.i(), y: r.i(), z: r.i()}
			idx[i] = specs[i].index()
		}
		name = "Slice"
		got, err = a.Slice(idx...)
		want = refSlice(a, specs)
	case 1:
		shape := make([]int, r.n(4))
		for i := range shape {
			shape[i] = r.i()
		}
		name = "Reshape"
		got, err = a.Reshape(shape...)
		want = refReshape(a, shape)
	case 2:
		name = "Transpose"
		got, err = a.Transpose(), nil
		want = refTranspose(a)
	case 3:
		name = "Squeeze"
		got, err = a.Squeeze()
		want = refSqueeze(a)
	case 4:
		ax := r.i()
		name = "ExpandDims"
		got, err = a.ExpandDims(ax)
		want = refExpandDims(a, ax)
	case 5:
		name = "Add"
		got, err = a.Add(b)
		want = refBroadcast(a, b, func(x, y float64) float64 { return x + y })
	case 6:
		name = "Mul"
		got, err = a.Mul(b)
		want = refBroadcast(a, b, func(x, y float64) float64 { return x * y })
	case 7:
		ax, keep := r.i(), r.n(2) == 1
		name = "SumAxis"
		got, err = a.SumAxis(ax, keep)
		want = refReduce(a, ax, keep, 0, func(acc, x float64) float64 { return acc + x }, true)
	case 8:
		ax, keep := r.i(), r.n(2) == 1
		name = "MaxAxis"
		got, err = a.MaxAxis(ax, keep)
		want = refReduce(a, ax, keep, math.Inf(-1), math.Max, false)
	case 9:
		idx := make([]int, r.n(4))
		for i := range idx {
			idx[i] = r.i()
		}
		name = "Take"
		got, err = a.Take(idx...)
		want = refTake(a, idx)
	case 10:
		ax := r.i()
		name = "Concatenate"
		got, err = Concatenate([]*Array{a, b}, ax)
		want = refConcat(a, b, ax)
	case 11:
		name = "MatMul"
		got, err = a.MatMul(b)
		want = refMatMul(a, b)
	case 12:
		name = "MaskSelect"
		got, err = a.MaskSelect(b)
		want = refMask(a, b)
	default:
		ax := r.i()
		name = "CumSum"
		got, err = a.CumSum(ax)
		want = refCumSum(a, ax)
	}
	switch {
	case want == nil && err == nil:
		t.Fatalf("%s(%v, %v): accepted, oracle rejects; got shape %v", name, a.Shape(), b.Shape(), got.Shape())
	case want != nil && err != nil:
		t.Fatalf("%s(%v, %v): %v; oracle gives shape %v", name, a.Shape(), b.Shape(), err, want.shape)
	case want == nil:
		return nil
	}
	checkAgainst(t, name, got, want)
	return got
}

// ref is an oracle result: a shape and its row-major values.
type ref struct {
	shape []int
	vals  []float64
}

// checkAgainst asserts got has want's shape and, read element by element
// through its own strides, want's values; and that Copy agrees.
func checkAgainst(t *testing.T, name string, got *Array, want *ref) {
	t.Helper()
	if !equalInts(got.Shape(), want.shape) {
		t.Fatalf("%s: shape %v, oracle %v", name, got.Shape(), want.shape)
	}
	if got.Size() != len(want.vals) {
		t.Fatalf("%s: Size %d, oracle %d", name, got.Size(), len(want.vals))
	}
	flat := got.Copy().contiguousData()
	k := 0
	eachIndex(want.shape, func(idx []int) {
		v := got.At(idx...)
		if !same(v, want.vals[k]) || !same(flat[k], want.vals[k]) {
			t.Fatalf("%s: At%v = %v, Copy = %v, oracle %v (shape %v)",
				name, idx, v, flat[k], want.vals[k], want.shape)
		}
		k++
	})
}

func same(x, y float64) bool { return x == y || (math.IsNaN(x) && math.IsNaN(y)) }

func equalInts(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

// eachIndex calls f with every index of shape in row-major order.
func eachIndex(shape []int, f func([]int)) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	idx := make([]int, len(shape))
	for k := 0; k < n; k++ {
		f(idx)
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
}

// build evaluates at(idx) for every index of shape.
func build(shape []int, at func([]int) float64) *ref {
	out := &ref{shape: append([]int{}, shape...)}
	eachIndex(shape, func(idx []int) { out.vals = append(out.vals, at(idx)) })
	return out
}

func values(a *Array) []float64 { return build(a.Shape(), func(i []int) float64 { return a.At(i...) }).vals }

// normAxis maps a possibly negative axis into [0, n), or -1 if out of range.
func normAxis(ax, n int) int {
	if ax < 0 {
		ax += n
	}
	if ax < 0 || ax >= n {
		return -1
	}
	return ax
}

// sliceSpec is one fuzzed index: kind 0 = A(x), 1 = Rng(x, y, z),
// 2 = Step(z), 3 = All().
type sliceSpec struct{ kind, x, y, z int }

func (s sliceSpec) index() Index {
	switch s.kind {
	case 0:
		return A(s.x)
	case 1:
		return Rng(s.x, s.y, s.z)
	case 2:
		return Step(s.z)
	}
	return All()
}

// positions is Python's slice.indices: the positions selected along an axis
// of length n, or ok=false for a zero step.
func (s sliceSpec) positions(n int) (pos []int, ok bool) {
	step := 1
	hasBounds := false
	switch s.kind {
	case 1:
		step, hasBounds = s.z, true
	case 2:
		step = s.z
	}
	if step == 0 {
		return nil, false
	}
	var start, stop int
	if step > 0 {
		start, stop = 0, n
	} else {
		start, stop = n-1, -1
	}
	if hasBounds {
		clamp := func(v int) int {
			if v < 0 {
				v += n
			}
			lo, hi := 0, n
			if step < 0 {
				lo, hi = -1, n-1
			}
			return min(max(v, lo), hi)
		}
		start, stop = clamp(s.x), clamp(s.y)
	}
	for i := start; (step > 0 && i < stop) || (step < 0 && i > stop); i += step {
		pos = append(pos, i)
	}
	return pos, true
}

func refSlice(a *Array, specs []sliceSpec) *ref {
	shape := a.Shape()
	if len(specs) > len(shape) {
		return nil
	}
	// Per input axis: either a fixed position (an integer index, axis dropped)
	// or the list of positions it ranges over.
	fixed := make([]int, len(shape))
	ranges := make([][]int, len(shape))
	var outShape []int
	var outAxes []int // input axis of each output axis
	for d, n := range shape {
		s := sliceSpec{kind: 3}
		if d < len(specs) {
			s = specs[d]
		}
		if s.kind == 0 {
			i := s.x
			if i < 0 {
				i += n
			}
			if i < 0 || i >= n {
				return nil
			}
			fixed[d], ranges[d] = i, nil
			continue
		}
		pos, ok := s.positions(n)
		if !ok {
			return nil
		}
		ranges[d] = pos
		outShape = append(outShape, len(pos))
		outAxes = append(outAxes, d)
	}
	if outShape == nil {
		outShape = []int{}
	}
	return build(outShape, func(idx []int) float64 {
		in := append([]int{}, fixed...)
		for o, d := range outAxes {
			in[d] = ranges[d][idx[o]]
		}
		return a.At(in...)
	})
}

func refReshape(a *Array, shape []int) *ref {
	n := a.Size()
	known, infer := 1, -1
	for i, d := range shape {
		switch {
		case d == -1 && infer < 0:
			infer = i
		case d < 0:
			return nil
		default:
			known *= d
		}
	}
	out := append([]int{}, shape...)
	if infer >= 0 {
		if known == 0 || n%known != 0 {
			return nil
		}
		out[infer] = n / known
	} else if known != n {
		return nil
	}
	return &ref{shape: out, vals: values(a)}
}

func refTranspose(a *Array) *ref {
	s := a.Shape()
	out := make([]int, len(s))
	for i := range s {
		out[i] = s[len(s)-1-i]
	}
	return build(out, func(idx []int) float64 {
		in := make([]int, len(idx))
		for i := range idx {
			in[i] = idx[len(idx)-1-i]
		}
		return a.At(in...)
	})
}

func refSqueeze(a *Array) *ref {
	out := []int{}
	for _, d := range a.Shape() {
		if d != 1 {
			out = append(out, d)
		}
	}
	return &ref{shape: out, vals: values(a)}
}

func refExpandDims(a *Array, ax int) *ref {
	s := a.Shape()
	if ax = normAxis(ax, len(s)+1); ax < 0 {
		return nil
	}
	out := append(append(append([]int{}, s[:ax]...), 1), s[ax:]...)
	return &ref{shape: out, vals: values(a)}
}

// refBroadcastShape is NumPy's rule: align on the right, each pair equal or 1.
func refBroadcastShape(x, y []int) ([]int, bool) {
	n := max(len(x), len(y))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		dx, dy := 1, 1
		if j := len(x) - n + i; j >= 0 {
			dx = x[j]
		}
		if j := len(y) - n + i; j >= 0 {
			dy = y[j]
		}
		switch {
		case dx == dy, dy == 1:
			out[i] = dx
		case dx == 1:
			out[i] = dy
		default:
			return nil, false
		}
	}
	return out, true
}

// atBroadcast reads a at an index of a broadcast shape of rank len(idx).
func atBroadcast(a *Array, idx []int) float64 {
	s := a.Shape()
	in := make([]int, len(s))
	for i := range s {
		if s[i] != 1 {
			in[i] = idx[len(idx)-len(s)+i]
		}
	}
	return a.At(in...)
}

func refBroadcast(a, b *Array, op func(x, y float64) float64) *ref {
	out, ok := refBroadcastShape(a.Shape(), b.Shape())
	if !ok {
		return nil
	}
	return build(out, func(idx []int) float64 { return op(atBroadcast(a, idx), atBroadcast(b, idx)) })
}

func refReduce(a *Array, ax int, keep bool, init float64, op func(acc, x float64) float64, hasIdentity bool) *ref {
	s := a.Shape()
	if ax = normAxis(ax, len(s)); ax < 0 {
		return nil
	}
	if s[ax] == 0 && !hasIdentity {
		return nil
	}
	var out []int
	for i, d := range s {
		switch {
		case i != ax:
			out = append(out, d)
		case keep:
			out = append(out, 1)
		}
	}
	if out == nil {
		out = []int{}
	}
	return build(out, func(idx []int) float64 {
		in := make([]int, 0, len(s))
		in = append(in, idx[:ax]...)
		in = append(in, 0)
		rest := idx[ax:]
		if keep {
			rest = idx[ax+1:]
		}
		in = append(in, rest...)
		acc := init
		for k := 0; k < s[ax]; k++ {
			in[ax] = k
			acc = op(acc, a.At(in...))
		}
		return acc
	})
}

func refTake(a *Array, idx []int) *ref {
	flat := values(a)
	out := &ref{shape: []int{len(idx)}}
	for _, i := range idx {
		if i < 0 {
			i += len(flat)
		}
		if i < 0 || i >= len(flat) {
			return nil
		}
		out.vals = append(out.vals, flat[i])
	}
	return out
}

func refConcat(a, b *Array, ax int) *ref {
	sa, sb := a.Shape(), b.Shape()
	if len(sa) != len(sb) || len(sa) == 0 {
		return nil
	}
	if ax = normAxis(ax, len(sa)); ax < 0 {
		return nil
	}
	for i := range sa {
		if i != ax && sa[i] != sb[i] {
			return nil
		}
	}
	out := append([]int{}, sa...)
	out[ax] += sb[ax]
	return build(out, func(idx []int) float64 {
		if idx[ax] < sa[ax] {
			return a.At(idx...)
		}
		in := append([]int{}, idx...)
		in[ax] -= sa[ax]
		return b.At(in...)
	})
}

func refMatMul(a, b *Array) *ref {
	sa, sb := a.Shape(), b.Shape()
	if len(sa) != 2 || len(sb) != 2 || sa[1] != sb[0] {
		return nil
	}
	return build([]int{sa[0], sb[1]}, func(idx []int) float64 {
		s := 0.0
		for k := 0; k < sa[1]; k++ {
			s += a.At(idx[0], k) * b.At(k, idx[1])
		}
		return s
	})
}

// refMask follows MaskSelect's contract: the mask must broadcast to a's shape
// without enlarging it (an extension of NumPy, which wants the same shape).
func refMask(a, m *Array) *ref {
	if s, ok := refBroadcastShape(a.Shape(), m.Shape()); !ok || !equalInts(s, a.Shape()) {
		return nil
	}
	out := &ref{}
	eachIndex(a.Shape(), func(idx []int) {
		if atBroadcast(m, idx) != 0 {
			out.vals = append(out.vals, a.At(idx...))
		}
	})
	out.shape = []int{len(out.vals)}
	return out
}

func refCumSum(a *Array, ax int) *ref {
	s := a.Shape()
	if ax = normAxis(ax, len(s)); ax < 0 {
		return nil
	}
	return build(s, func(idx []int) float64 {
		in := append([]int{}, idx...)
		acc := 0.0
		for k := 0; k <= idx[ax]; k++ {
			in[ax] = k
			acc += a.At(in...)
		}
		return acc
	})
}

// byteReader hands out small integers from the fuzz input, and zeros once it
// is exhausted.
type byteReader struct {
	b []byte
	p int
}

func (r *byteReader) more() bool { return r.p < len(r.b) }

func (r *byteReader) next() byte {
	if r.p >= len(r.b) {
		return 0
	}
	v := r.b[r.p]
	r.p++
	return v
}

// n returns a value in [0, m).
func (r *byteReader) n(m int) int { return int(r.next()) % m }

// i returns a small signed value in [-8, 8), the range that exercises
// negative indices, out-of-range bounds and reversed steps.
func (r *byteReader) i() int { return int(r.next()%16) - 8 }

// array builds an array of rank 0-3 with dimensions in [-1, 4] and small
// integer values, or nil when the shape is (correctly) rejected.
func (r *byteReader) array(t *testing.T) *Array {
	shape := make([]int, r.n(4))
	for i := range shape {
		shape[i] = r.n(6) - 1
	}
	a, err := New(shape...)
	if err != nil {
		for _, d := range shape {
			if d < 0 {
				return nil
			}
		}
		t.Fatalf("New(%v): %v", shape, err)
	}
	for i := range a.data {
		a.data[i] = float64(int(r.next())%7 - 3)
	}
	return a
}
