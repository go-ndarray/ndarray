package ndarray

import "testing"

// generalMaterialize is the original per-element odometer gather, kept here as
// the reference the fast-path materialize() must agree with for every view.
func (a *Array) generalMaterialize() []float64 {
	out := make([]float64, a.Size())
	a.forEach(func(linear, pos int) {
		out[linear] = a.data[pos]
	})
	return out
}

// TestMaterializeFastPaths exercises every branch of the bulk-copy materialize:
// the fully-contiguous copy, the unit-stride-inner row copy (a row slice of a
// matrix), the length-1 inner edge case, the strided-inner fallback (a
// transpose), and the empty array. Each must match the element-wise reference.
func TestMaterializeFastPaths(t *testing.T) {
	base := mustArr(t, ok(Arange(0, 24, 1))) // 0..23
	m := mustArr(t, ok(base.Reshape(4, 6)))

	cases := map[string]*Array{
		"contiguous":      m,
		"row-slice":       mustArr(t, ok(m.Slice(R(1, 3), All()))), // rows 1..2, inner stride 1
		"col-slice":       mustArr(t, ok(m.Slice(All(), R(2, 5)))), // strided rows but inner stride 1
		"single-col":      mustArr(t, ok(m.Slice(All(), A(2)))),    // drops axis -> 1-D contiguous
		"transpose":       m.Transpose(),                           // inner axis strided -> fallback
		"step-rows":       mustArr(t, ok(m.Slice(Step(2), All()))), // every other row, inner stride 1
		"reversed-cols":   mustArr(t, ok(m.Slice(All(), Step(-1)))),
		"empty-row-slice": mustArr(t, ok(m.Slice(R(2, 2), All()))),
	}
	for name, v := range cases {
		got := v.materialize()
		want := v.generalMaterialize()
		if len(got) != len(want) {
			t.Fatalf("%s: len %d != %d", name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s [%d]: %v != %v\n got=%v\nwant=%v", name, i, got[i], want[i], got, want)
			}
		}
	}
}

// generalBroadcast is the original per-element odometer broadcast, the reference
// the fast-path broadcastTo() must reproduce.
func (a *Array) generalBroadcast(shape []int) []float64 {
	n := len(shape)
	estrides := make([]int, n)
	for i := 0; i < n; i++ {
		j := len(a.shape) - n + i
		if j < 0 || a.shape[j] == 1 {
			estrides[i] = 0
		} else {
			estrides[i] = a.strides[j]
		}
	}
	out := make([]float64, prod(shape))
	if len(out) == 0 {
		return out
	}
	idx := make([]int, n)
	for linear := range out {
		pos := a.offset
		for axis, i := range idx {
			pos += i * estrides[axis]
		}
		out[linear] = a.data[pos]
		for axis := n - 1; axis >= 0; axis-- {
			idx[axis]++
			if idx[axis] < shape[axis] {
				break
			}
			idx[axis] = 0
		}
	}
	return out
}

// TestBroadcastToFastPaths exercises the three branches of broadcastTo: a real
// unit-stride innermost axis (row -> matrix), a broadcast length-1 innermost
// axis splatted per run (column -> matrix), and the strided-inner fallback (a
// transposed operand broadcast along a new outer axis). Each is checked against
// the element-wise reference, and the empty target is checked too.
func TestBroadcastToFastPaths(t *testing.T) {
	row, _ := FromData([]float64{1, 2, 3}, 1, 3) // last axis real, stride 1
	col, _ := FromData([]float64{1, 2, 3}, 3, 1) // last axis length 1 -> splat
	colT := col.Transpose()                      // (1,3) view
	mtx, _ := FromData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	trans := mtx.Transpose() // (3,2) view whose last axis has stride 3 -> fallback

	type tc struct {
		a     *Array
		shape []int
	}
	cases := map[string]tc{
		"row->matrix":     {row, []int{4, 3}},
		"col->matrix":     {col, []int{3, 4}},
		"colT":            {colT, []int{2, 3}},
		"transpose-outer": {trans, []int{5, 3, 2}}, // prepend an outer broadcast axis
		"empty":           {row, []int{0, 3}},
	}
	for name, c := range cases {
		got := c.a.broadcastTo(c.shape)
		want := c.a.generalBroadcast(c.shape)
		if len(got) != len(want) {
			t.Fatalf("%s: len %d != %d", name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s [%d]: %v != %v\n got=%v\nwant=%v", name, i, got[i], want[i], got, want)
			}
		}
	}
}
