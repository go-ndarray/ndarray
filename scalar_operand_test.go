package ndarray

import (
	"math"
	"testing"
)

// TestScalarOperandPaths: the operations that take binOp's single-element
// path (#118) — a scalar, a 1×1 array, a one-element view with an offset, on
// either side — give the same bits as the element-by-element definition.
func TestScalarOperandPaths(t *testing.T) {
	const n = 1001
	x, _ := Linspace(-3, 5, n)
	xs := x.data // Linspace is contiguous at offset 0
	mat, _ := x.Reshape(7, 143)
	big, _ := Linspace(0, 10, 11)
	view, _ := big.Slice(R(4, 5)) // the element 4.0, at offset 4
	oneByOne, _ := Full(4, 1, 1)

	ops := []struct {
		name string
		got  func() (*Array, error)
		want func(v float64) float64
	}{
		{"AddScalar", func() (*Array, error) { return x.AddScalar(0.1), nil }, func(v float64) float64 { return v + 0.1 }},
		{"SubScalar", func() (*Array, error) { return x.SubScalar(0.1), nil }, func(v float64) float64 { return v - 0.1 }},
		{"MulScalar", func() (*Array, error) { return x.MulScalar(0.1), nil }, func(v float64) float64 { return v * 0.1 }},
		{"DivScalar", func() (*Array, error) { return x.DivScalar(0.1), nil }, func(v float64) float64 { return v / 0.1 }},
		{"scalar-first Div", func() (*Array, error) { return scalarArray(0.1).Div(x) }, func(v float64) float64 { return 0.1 / v }},
		{"scalar-first Sub", func() (*Array, error) { return scalarArray(0.1).Sub(x) }, func(v float64) float64 { return 0.1 - v }},
		{"view at offset", func() (*Array, error) { return x.Sub(view) }, func(v float64) float64 { return v - 4 }},
		{"view first", func() (*Array, error) { return view.Div(x) }, func(v float64) float64 { return 4 / v }},
		{"1x1 against matrix", func() (*Array, error) { return mat.Mul(oneByOne) }, func(v float64) float64 { return v * 4 }},
		{"DivInto scalar first", func() (*Array, error) {
			out, _ := New(n)
			return out, scalarArray(0.1).DivInto(out, x)
		}, func(v float64) float64 { return 0.1 / v }},
		{"SubInto aliasing x", func() (*Array, error) {
			y := x.Copy()
			return y, y.SubInto(y, scalarArray(0.1))
		}, func(v float64) float64 { return v - 0.1 }},
	}
	for _, op := range ops {
		r, err := op.got()
		if err != nil {
			t.Fatalf("%s: %v", op.name, err)
		}
		if r.Size() != n {
			t.Fatalf("%s: size %d, want %d", op.name, r.Size(), n)
		}
		for i, g := range r.data[r.offset : r.offset+n] {
			if w := op.want(xs[i]); math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("%s [%d]: got %v, want %v", op.name, i, g, w)
			}
		}
	}
	if got := mat.MulScalar(2).Shape(); got[0] != 7 || got[1] != 143 {
		t.Fatalf("shape %v, want [7 143]", got)
	}
}

// TestScalarOperandAllocs: a scalar operation allocates its result and a
// block of the scalar, never a copy of the scalar at the result's size.
func TestScalarOperandAllocs(t *testing.T) {
	x, _ := Linspace(0, 1, 100_000)
	var r *Array
	bytes := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			r = x.AddScalar(1.5)
		}
	}).AllocedBytesPerOp()
	if limit := int64(8*100_000 + 8*1024); bytes > limit {
		t.Fatalf("AddScalar allocates %d B per call, want at most %d (result + one block)", bytes, limit)
	}
	_ = r
	out, _ := New(100_000)
	intoBytes := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			_ = x.AddInto(out, scalarArray(1.5))
		}
	}).AllocedBytesPerOp()
	if intoBytes > 8*1024 {
		t.Fatalf("AddInto with a scalar allocates %d B per call, want at most one block", intoBytes)
	}
}
