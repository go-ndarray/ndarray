package ndarray

import (
	"errors"
	"math"
	"math/cmplx"
	"reflect"
	"strings"
	"testing"
)

// mustPanic fails the test unless f panics with a message containing want.
func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		p := recover()
		if p == nil {
			t.Errorf("no panic, want %q", want)
			return
		}
		if s, _ := p.(string); !strings.Contains(s, want) {
			t.Errorf("panic %v, want %q", p, want)
		}
	}()
	f()
}

func data[T elemT](t *testing.T, a *Array) []T {
	t.Helper()
	d, ok := Data[T](a)
	if !ok {
		t.Fatalf("Data[%v] of a %v array", dtypeOf[T](), a.DType())
	}
	return d
}

func TestTypedConstructionAndAccess(t *testing.T) {
	a, err := FromSlice([]int16{1, -2, 3, 4, 5, 6}, 2, 3)
	if err != nil || a.DType() != Int16 {
		t.Fatal(a, err)
	}
	if got := Item[int16](a, 0, 1); got != -2 {
		t.Errorf("Item = %d", got)
	}
	if got := a.At(1, 2); got != 6 {
		t.Errorf("At = %g", got)
	}
	a.Set(-7.9, 1, 0) // truncates toward zero
	if got := Item[int16](a, 1, 0); got != -7 {
		t.Errorf("Set(-7.9) stored %d", got)
	}
	a.SetComplex(9+5i, 0, 0)
	if got := a.AtComplex(0, 0); got != 9 {
		t.Errorf("SetComplex/AtComplex = %v", got)
	}
	if _, ok := Data[float32](a); ok {
		t.Error("Data[float32] of an int16 array succeeded")
	}
	mustPanic(t, "Item[int8] of a int16", func() { Item[int8](a, 0, 0) })

	// Data of a strided view is a contiguous copy.
	col, _ := a.Slice(All(), A(1))
	if got := data[int16](t, col); !reflect.DeepEqual(got, []int16{-2, 5}) {
		t.Errorf("column = %v", got)
	}

	if _, err := FromSlice([]bool{true}, 2); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("FromSlice size mismatch err = %v", err)
	}
	if _, err := FromSlice([]bool{true}, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("FromSlice negative shape err = %v", err)
	}
	if _, err := ZerosOf(DType(99), 2); !errors.Is(err, ErrDType) {
		t.Errorf("ZerosOf(99) err = %v", err)
	}
	if _, err := ZerosOf(Int8, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("ZerosOf negative err = %v", err)
	}
	if _, err := FullOf(Int8, 1, -1); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("FullOf negative err = %v", err)
	}
	f, _ := ZerosOf(Float64, 2)
	if f.DType() != Float64 || f.Size() != 2 {
		t.Errorf("ZerosOf(Float64) = %v", f)
	}
	c, _ := FullOf(Complex64, 1-2i, 2)
	if got := data[complex64](t, c); got[1] != 1-2i {
		t.Errorf("FullOf complex = %v", got)
	}
	mustPanic(t, "use AtComplex", func() { c.At(0) })
	for _, typ := range []any{[]bool{}, []int8{}, []uint32{}, []complex128{}} {
		r := fromStore(typ, []int{0})
		if r.dtype != dtypeOfStore(typ) {
			t.Errorf("fromStore(%T) dtype %v", typ, r.dtype)
		}
	}
	if dtypeOf[float64]() != Float64 || dtypeOf[uint16]() != Uint16 {
		t.Error("dtypeOf")
	}
}

func TestTypedString(t *testing.T) {
	cases := []struct {
		a    *Array
		want string
	}{
		{must(FromSlice([]bool{true, false}, 2)), "Array(shape=[2], dtype=bool, data=[true false])"},
		{must(FromSlice([]int8{-1, 2}, 2)), "Array(shape=[2], dtype=int8, data=[-1 2])"},
		{must(FromSlice([]uint64{math.MaxUint64}, 1)), "Array(shape=[1], dtype=uint64, data=[18446744073709551615])"},
		{must(FromSlice([]float32{0.1}, 1)), "Array(shape=[1], dtype=float32, data=[0.1])"},
		{must(FromSlice([]complex64{1 - 2i, complex(0, float32(math.Copysign(0, -1)))}, 2)),
			"Array(shape=[2], dtype=complex64, data=[(1-2j) (0-0j)])"},
		{must(FromSlice([]complex128{0.5 + 1i}, 1)), "Array(shape=[1], dtype=complex128, data=[(0.5+1j)])"},
	}
	for _, c := range cases {
		if got := c.a.String(); got != c.want {
			t.Errorf("String = %s, want %s", got, c.want)
		}
	}
}

func must(a *Array, err error) *Array {
	if err != nil {
		panic(err)
	}
	return a
}

func TestTypedStructural(t *testing.T) {
	a := must(FromSlice([]int32{1, 2, 3, 4, 5, 6}, 2, 3))
	tr := a.Transpose()
	if tr.DType() != Int32 || Item[int32](tr, 2, 1) != 6 {
		t.Errorf("Transpose = %v", tr)
	}
	// Reshape of a non-contiguous view copies.
	r := must(tr.Reshape(6))
	if got := data[int32](t, r); !reflect.DeepEqual(got, []int32{1, 4, 2, 5, 3, 6}) {
		t.Errorf("Reshape of transpose = %v", got)
	}
	if got := data[int32](t, tr.Ravel()); !reflect.DeepEqual(got, []int32{1, 4, 2, 5, 3, 6}) {
		t.Errorf("Ravel = %v", got)
	}
	fl := a.Flatten()
	fl.Set(0, 0)
	if Item[int32](a, 0, 0) != 1 {
		t.Error("Flatten shares storage")
	}
	cp := tr.Copy()
	if cp.DType() != Int32 || !cp.isContiguous() {
		t.Errorf("Copy = %v", cp)
	}
	row := must(a.Slice(A(1)))
	ex := must(row.ExpandDims(0))
	sq := must(ex.Squeeze())
	if got := data[int32](t, sq); !reflect.DeepEqual(got, []int32{4, 5, 6}) {
		t.Errorf("slice/expand/squeeze = %v", got)
	}

	// Concatenation promotes, like numpy.concatenate.
	b := must(FromSlice([]float32{0.5, 1.5, 2.5}, 1, 3))
	cat := must(Concatenate([]*Array{a, b}, 0))
	if cat.DType() != Float64 || cat.At(2, 2) != 2.5 || cat.At(1, 0) != 4 {
		t.Errorf("Concatenate int32+float32 = %v", cat)
	}
	u8 := must(FromSlice([]uint8{1, 2, 3}, 1, 3))
	cat = must(VStack([]*Array{u8, must(FromSlice([]int8{-1, -2, -3}, 3))}))
	if cat.DType() != Int16 || Item[int16](cat, 1, 2) != -3 {
		t.Errorf("VStack uint8+int8 = %v", cat)
	}
	if e := must(Concatenate([]*Array{must(ZerosOf(Int8, 0, 2)), must(ZerosOf(Int8, 0, 2))}, 0)); e.Size() != 0 || e.DType() != Int8 {
		t.Errorf("empty concat = %v", e)
	}
	st := must(Stack([]*Array{must(FromSlice([]bool{true}, 1)), must(FromSlice([]bool{false}, 1))}, 1))
	if st.DType() != Bool || !reflect.DeepEqual(st.Shape(), []int{1, 2}) {
		t.Errorf("Stack bool = %v", st)
	}

	tk := must(a.Take(-1, 0))
	if got := data[int32](t, tk); !reflect.DeepEqual(got, []int32{6, 1}) {
		t.Errorf("Take = %v", got)
	}
	if _, err := a.Take(6); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("Take out of range err = %v", err)
	}
}

func TestMasks(t *testing.T) {
	x := must(FromSlice([]int64{5, -1, 7, 0}, 4))
	m := must(x.Greater(Scalar(0)))
	if m.DType() != Bool || !reflect.DeepEqual(data[bool](t, m), []bool{true, false, true, false}) {
		t.Fatalf("x > 0 = %v", m)
	}
	sel := must(x.MaskSelect(m))
	if got := data[int64](t, sel); !reflect.DeepEqual(got, []int64{5, 7}) {
		t.Errorf("x[x>0] = %v", got)
	}
	// A float array and a bool mask, and the old 0/1 float mask.
	f := must(FromData([]float64{1, 2, 3, 4}, 4))
	if got := must(f.MaskSelect(m)).materialize(); !reflect.DeepEqual(got, []float64{1, 3}) {
		t.Errorf("float[boolmask] = %v", got)
	}
	if got := data[int64](t, must(x.MaskSelect(must(FromData([]float64{0, 1, 0, 1}, 4))))); !reflect.DeepEqual(got, []int64{-1, 0}) {
		t.Errorf("int[floatmask] = %v", got)
	}
	nz := m.Nonzero()
	if got := data[int64](t, nz); !reflect.DeepEqual(got, []int64{0, 2}) {
		t.Errorf("Nonzero = %v", got)
	}
	not := m.LogicalNot()
	if got := data[bool](t, not); !reflect.DeepEqual(got, []bool{false, true, false, true}) {
		t.Errorf("LogicalNot = %v", got)
	}
	y := must(FromSlice([]float64{1, 1, 0, 0}, 4))
	for name, c := range map[string]struct {
		f    func(*Array) (*Array, error)
		want []bool
	}{
		"and": {m.LogicalAnd, []bool{true, false, false, false}},
		"or":  {m.LogicalOr, []bool{true, true, true, false}},
		"xor": {m.LogicalXor, []bool{false, true, true, false}},
	} {
		if got := data[bool](t, must(c.f(y))); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s = %v, want %v", name, got, c.want)
		}
	}
	if _, err := m.LogicalAnd(must(Zeros(3))); !errors.Is(err, ErrBroadcast) {
		t.Errorf("LogicalAnd broadcast err = %v", err)
	}
	if !m.Any() || m.All() || must(Zeros(2)).Any() || !must(ZerosOf(Bool, 0)).All() || !must(OnesOf(Int8, 2)).All() {
		t.Error("Any/All")
	}
	if _, err := x.Equal(must(Zeros(3))); !errors.Is(err, ErrBroadcast) {
		t.Errorf("compare broadcast err = %v", err)
	}
}

func TestTypedArithmeticEdges(t *testing.T) {
	b := must(FromSlice([]bool{true, false}, 2))
	if _, err := b.Sub(b); !errors.Is(err, ErrDType) {
		t.Errorf("bool - bool err = %v", err)
	}
	i := must(FromSlice([]int8{1, 2}, 2))
	if _, err := i.Add(must(ZerosOf(Int8, 3))); !errors.Is(err, ErrBroadcast) {
		t.Errorf("broadcast err = %v", err)
	}
	// Into: converted into out's dtype, with the shape and contiguity checks.
	out := must(ZerosOf(Int16, 2))
	if err := i.AddInto(out, i); err != nil || !reflect.DeepEqual(data[int16](t, out), []int16{2, 4}) {
		t.Errorf("AddInto = %v, %v", out, err)
	}
	for _, f := range []func(out, b *Array) error{i.SubInto, i.MulInto, i.DivInto} {
		if err := f(out, i); err != nil {
			t.Error(err)
		}
	}
	if err := i.AddInto(out, must(ZerosOf(Int8, 3))); !errors.Is(err, ErrBroadcast) {
		t.Errorf("AddInto broadcast err = %v", err)
	}
	if err := i.AddInto(must(ZerosOf(Int16, 3)), i); !errors.Is(err, ErrBroadcast) {
		t.Errorf("AddInto shape err = %v", err)
	}
	strided := must(must(ZerosOf(Int16, 4)).Slice(Step(2)))
	if err := i.AddInto(strided, i); !errors.Is(err, ErrBroadcast) {
		t.Errorf("AddInto strided err = %v", err)
	}
	if err := i.SqrtInto(must(Zeros(2))); !errors.Is(err, ErrDType) {
		t.Errorf("SqrtInto int err = %v", err)
	}
	// Float64 result from an int: the float path runs.
	f := must(FromData([]float64{0.5, 0.25}, 2))
	if got := must(i.Mul(f)).materialize(); !reflect.DeepEqual(got, []float64{0.5, 0.5}) {
		t.Errorf("int8 * float64 = %v", got)
	}
}

func TestTypedUnary(t *testing.T) {
	i := must(FromSlice([]int32{1, 4, -9}, 3))
	for name, f := range map[string]func() *Array{
		"log2": i.Log2, "log10": i.Log10, "cos": i.Cos, "tan": i.Tan,
	} {
		if r := f(); r.DType() != Float64 || r.Size() != 3 {
			t.Errorf("%s = %v", name, r)
		}
	}
	for name, f := range map[string]func() *Array{"ceil": i.Ceil, "round": i.Round} {
		if got := data[int32](t, f()); !reflect.DeepEqual(got, []int32{1, 4, -9}) {
			t.Errorf("%s = %v", name, got)
		}
	}
	h := must(FromSlice([]float32{1.5, -2.5}, 2))
	if got := data[float32](t, h.Ceil()); !reflect.DeepEqual(got, []float32{2, -2}) {
		t.Errorf("float32 ceil = %v", got)
	}
	if got := data[float32](t, h.Round()); !reflect.DeepEqual(got, []float32{2, -3}) {
		t.Errorf("float32 round = %v", got)
	}
	if got := data[float32](t, h.Power(2)); !reflect.DeepEqual(got, []float32{2.25, 6.25}) {
		t.Errorf("float32 power = %v", got)
	}
	if r := i.Power(0.5); r.DType() != Float64 || r.At(1) != 2 {
		t.Errorf("int32 power = %v", r)
	}
	c := must(FromSlice([]complex128{3 + 4i}, 1))
	if got := c.Round(); got.AtComplex(0) != 3+4i {
		t.Errorf("complex round = %v", got)
	}
	mustPanic(t, "not defined for complex128", func() { c.Ceil() })
	if r := i.Map(func(x float64) float64 { return x + 0.5 }); r.DType() != Float64 || r.At(0) != 1.5 {
		t.Errorf("int Map = %v", r)
	}
	mustPanic(t, "Map of a complex128", func() { c.Map(math.Abs) })
	mustPanic(t, "negating a bool", func() { must(OnesOf(Bool, 1)).Neg() })
	if got := data[uint8](t, must(FromSlice([]uint8{200}, 1)).Neg()); got[0] != 56 {
		t.Errorf("-uint8(200) = %v", got)
	}
}

func TestTypedReductions(t *testing.T) {
	big := must(FromSlice([]int64{1 << 60, 1, 1}, 3))
	if s := big.SumAll(); Item[int64](s) != 1<<60+2 || s.Ndim() != 0 {
		t.Errorf("SumAll int64 = %v", s)
	}
	if p := must(FromSlice([]uint8{200, 2}, 2)).ProdAll(); Item[uint64](p) != 400 {
		t.Errorf("ProdAll uint8 = %v", p)
	}
	f := must(FromData([]float64{1, 2, 3}, 3))
	if f.SumAll().At() != 6 || f.ProdAll().At() != 6 {
		t.Error("SumAll/ProdAll float64")
	}
	i := must(FromSlice([]int16{3, -1, 4}, 3))
	if i.Sum() != 6 || i.Prod() != -12 {
		t.Errorf("Sum/Prod = %g %g", i.Sum(), i.Prod())
	}
	if m, err := i.Mean(); err != nil || m != 2 {
		t.Errorf("Mean = %g %v", m, err)
	}
	if m, err := i.Max(); err != nil || m != 4 {
		t.Errorf("Max = %g %v", m, err)
	}
	if m, err := i.Min(); err != nil || m != -1 {
		t.Errorf("Min = %g %v", m, err)
	}
	if k, err := i.ArgMax(); err != nil || k != 2 {
		t.Errorf("ArgMax = %d %v", k, err)
	}
	if k, err := i.ArgMin(); err != nil || k != 1 {
		t.Errorf("ArgMin = %d %v", k, err)
	}
	if got := data[int64](t, i.CumSumFlat()); !reflect.DeepEqual(got, []int64{3, 2, 6}) {
		t.Errorf("CumSumFlat = %v", got)
	}
	if got := data[int64](t, i.CumProdFlat()); !reflect.DeepEqual(got, []int64{3, -3, -12}) {
		t.Errorf("CumProdFlat = %v", got)
	}
	c := must(FromSlice([]complex64{1 + 1i, 2}, 2))
	if s := c.SumAll(); Item[complex64](s) != 3+1i {
		t.Errorf("complex SumAll = %v", s)
	}
	for name, f := range map[string]func(){
		"Sum":  func() { c.Sum() },
		"Prod": func() { c.Prod() },
		"Mean": func() { _, _ = c.Mean() },
		"Max":  func() { _, _ = c.Max() },
		"Min":  func() { _, _ = c.Min() },
	} {
		mustPanic(t, name+" of a complex64", f)
	}

	// Axis errors and empty axes.
	if _, err := i.SumAxis(1, false); !errors.Is(err, ErrAxis) {
		t.Errorf("SumAxis(1) err = %v", err)
	}
	if _, err := i.ArgMaxAxis(1, false); !errors.Is(err, ErrAxis) {
		t.Errorf("ArgMaxAxis(1) err = %v", err)
	}
	if _, err := i.CumSum(1); !errors.Is(err, ErrAxis) {
		t.Errorf("CumSum(1) err = %v", err)
	}
	e := must(ZerosOf(Int8, 2, 0))
	if s := must(e.SumAxis(1, false)); !reflect.DeepEqual(data[int64](t, s), []int64{0, 0}) {
		t.Errorf("empty SumAxis = %v", s)
	}
	if p := must(e.ProdAxis(1, true)); !reflect.DeepEqual(data[int64](t, p), []int64{1, 1}) {
		t.Errorf("empty ProdAxis = %v", p)
	}
	if _, err := e.MaxAxis(1, false); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("empty MaxAxis err = %v", err)
	}
	if _, err := e.ArgMinAxis(1, false); !errors.Is(err, ErrShapeMismatch) {
		t.Errorf("empty ArgMinAxis err = %v", err)
	}
	if s := must(e.CumSum(1)); s.Size() != 0 || s.DType() != Int64 {
		t.Errorf("empty CumSum = %v", s)
	}
	if s := must(must(ZerosOf(Int8, 0, 3)).MinAxis(1, false)); s.Size() != 0 {
		t.Errorf("MinAxis of (0,3) = %v", s)
	}
	if s := must(must(ZerosOf(Int8, 0, 3)).ArgMaxAxis(1, false)); s.Size() != 0 {
		t.Errorf("ArgMaxAxis of (0,3) = %v", s)
	}
}

func TestTypedClipWhere(t *testing.T) {
	i := must(FromSlice([]int8{-5, 0, 5}, 3))
	if r := must(i.Clip(-1, 1)); r.DType() != Float64 || !reflect.DeepEqual(r.materialize(), []float64{-1, 0, 1}) {
		t.Errorf("int8 Clip = %v", r)
	}
	h := must(FromSlice([]float32{-5, float32(math.NaN()), 5}, 3))
	r := must(h.Clip(-1, 1))
	if got := data[float32](t, r); got[0] != -1 || got[2] != 1 || got[1] == got[1] {
		t.Errorf("float32 Clip = %v", got)
	}
	if _, err := must(OnesOf(Complex64, 1)).Clip(0, 1); !errors.Is(err, ErrDType) {
		t.Errorf("complex Clip err = %v", err)
	}
	w := must(Where(must(FromSlice([]bool{true, false, true}, 3)), i, Scalar(9)))
	if got := data[int8](t, w); !reflect.DeepEqual(got, []int8{-5, 9, 5}) {
		t.Errorf("Where = %v", got)
	}
}

func TestComplexParts(t *testing.T) {
	c := must(FromSlice([]complex64{3 + 4i, -1 - 1i}, 2))
	if got := data[float32](t, c.Real()); !reflect.DeepEqual(got, []float32{3, -1}) {
		t.Errorf("Real = %v", got)
	}
	if got := data[float32](t, c.Imag()); !reflect.DeepEqual(got, []float32{4, -1}) {
		t.Errorf("Imag = %v", got)
	}
	if got := data[complex64](t, c.Conj()); !reflect.DeepEqual(got, []complex64{3 - 4i, -1 + 1i}) {
		t.Errorf("Conj = %v", got)
	}
	if got := data[float32](t, c.Abs()); !reflect.DeepEqual(got, []float32{5, float32(math.Sqrt2)}) {
		t.Errorf("Abs = %v", got)
	}
	an := data[float32](t, c.Angle())
	if math.Abs(float64(an[1])+3*math.Pi/4) > 1e-6 {
		t.Errorf("Angle = %v", an)
	}
	z := must(FromSlice([]complex128{1i}, 1))
	if got := data[float64](t, z.Angle()); got[0] != math.Pi/2 {
		t.Errorf("Angle complex128 = %v", got)
	}
	if got := data[float64](t, z.Abs()); got[0] != 1 {
		t.Errorf("Abs complex128 = %v", got)
	}
	if got := data[complex128](t, z.Conj()); got[0] != cmplx.Conj(1i) {
		t.Errorf("Conj complex128 = %v", got)
	}
	r := must(FromSlice([]int8{-2}, 1))
	if got := data[int8](t, r.Real()); got[0] != -2 {
		t.Errorf("Real of int8 = %v", got)
	}
	if got := data[int8](t, r.Imag()); got[0] != 0 {
		t.Errorf("Imag of int8 = %v", got)
	}
	if got := data[int8](t, r.Conj()); got[0] != -2 {
		t.Errorf("Conj of int8 = %v", got)
	}
	if got := data[float64](t, r.Angle()); got[0] != math.Pi {
		t.Errorf("Angle of int8 = %v", got)
	}
}

func TestTypedLinalg(t *testing.T) {
	a := must(FromSlice([]int32{1, 2, 3, 4}, 2, 2))
	v := must(FromSlice([]int32{1, -1}, 2))
	cases := []struct {
		x, y  *Array
		shape []int
		want  []int32
	}{
		{v, v, []int{}, []int32{2}},
		{a, v, []int{2}, []int32{-1, -1}},
		{v, a, []int{2}, []int32{-2, -2}},
		{a, a, []int{2, 2}, []int32{7, 10, 15, 22}},
	}
	for _, c := range cases {
		r, err := c.x.Dot(c.y)
		if err != nil || !sameShape(r.Shape(), c.shape) || !reflect.DeepEqual(data[int32](t, r), c.want) {
			t.Errorf("Dot %v·%v = %v, %v", c.x.Shape(), c.y.Shape(), r, err)
		}
	}
	if _, err := v.Dot(must(ZerosOf(Int32, 3))); !errors.Is(err, ErrLinalg) {
		t.Errorf("Dot mismatch err = %v", err)
	}
	f := must(FromData([]float64{0.5, 0.5}, 2, 1))
	if r := must(a.MatMul(f)); r.DType() != Float64 || !reflect.DeepEqual(r.materialize(), []float64{1.5, 3.5}) {
		t.Errorf("int32 @ float64 = %v", r)
	}
	b := must(FromSlice([]bool{true, false, false, true}, 2, 2))
	if r := must(b.MatMul(b)); !reflect.DeepEqual(data[bool](t, r), []bool{true, false, false, true}) {
		t.Errorf("bool matmul = %v", r)
	}
	o := v.Outer(must(FromSlice([]int8{2, 3}, 2)))
	if o.DType() != Int32 || !reflect.DeepEqual(data[int32](t, o), []int32{2, 3, -2, -3}) {
		t.Errorf("Outer = %v", o)
	}
}

// TestGeneratedDispatchRejectsForeignValues covers the guards at the end of
// the generated type switches: they are only reached by a programming error.
func TestGeneratedDispatchRejectsForeignValues(t *testing.T) {
	bad := []string{"x"}
	for name, f := range map[string]func(){
		"dtypeOfStore":   func() { dtypeOfStore(bad) },
		"makeStore":      func() { makeStore(DType(99), 1) },
		"storeLen":       func() { storeLen(bad) },
		"cloneStore":     func() { cloneStore(bad) },
		"takeStore":      func() { takeStore(bad, nil) },
		"gatherStore":    func() { gatherStore(&Array{ext: bad, dtype: Int8}) },
		"broadcastStore": func() { broadcastStore(&Array{ext: bad, dtype: Int8}, nil) },
		"elemComplex":    func() { elemComplex(bad, 0) },
		"elemString":     func() { elemString(bad, 0) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s accepted a []string", name)
				}
			}()
			f()
		}()
	}
}

// TestEveryDTypeThroughTheGenericPaths runs the shape-only operations,
// element access and formatting on an array of every dtype, so each case of
// the generated type switches is taken.
func TestEveryDTypeThroughTheGenericPaths(t *testing.T) {
	for _, dt := range allDTypes() {
		a := must(ZerosOf(dt, 2, 3))
		a.Set(1, 0, 1)
		a.SetComplex(1, 1, 2)
		if !dt.IsComplex() && a.At(0, 1) != 1 {
			t.Errorf("%v: Set/At", dt)
		}
		if a.AtComplex(1, 2) != 1 {
			t.Errorf("%v: SetComplex/AtComplex = %v", dt, a.AtComplex(1, 2))
		}
		tr := a.Transpose()
		if got := tr.Copy(); got.DType() != dt || got.AtComplex(2, 1) != 1 {
			t.Errorf("%v: transposed copy = %v", dt, got)
		}
		if got := must(Concatenate([]*Array{a, a}, 1)); got.DType() != dt || got.AtComplex(1, 5) != 1 {
			t.Errorf("%v: concatenate = %v", dt, got)
		}
		if got := must(a.Take(1, 5)); got.AtComplex(1) != 1 {
			t.Errorf("%v: take = %v", dt, got)
		}
		if !strings.Contains(a.String(), "shape=[2 3]") {
			t.Errorf("%v: String = %s", dt, a)
		}
		empty := must(ZerosOf(dt, 0, 3)).Transpose().Copy()
		if empty.Size() != 0 {
			t.Errorf("%v: empty copy = %v", dt, empty)
		}
		for _, to := range allDTypes() {
			src, dst := makeStore(dt, 1), makeStore(to, 1)
			setComplex(src, 0, 1)
			convertStore(dst, src)
			if elemComplex(dst, 0) != 1 {
				t.Errorf("convert %v -> %v of 1 = %v", dt, to, elemComplex(dst, 0))
			}
		}
	}
	for _, d := range []any{[]int8{1}, []int16{1}, []int32{1}, []int64{1}, []uint8{1}, []uint16{1},
		[]uint32{1}, []uint64{1}, []float32{1}, []float64{1}, []complex64{1}, []complex128{1}, []bool{true}} {
		a := fromStore(d, []int{1})
		switch a.dtype {
		case Int8:
			_ = data[int8](t, a)
		case Int16:
			_ = data[int16](t, a)
		case Int32:
			_ = data[int32](t, a)
		case Int64:
			_ = data[int64](t, a)
		case Uint8:
			_ = data[uint8](t, a)
		case Uint16:
			_ = data[uint16](t, a)
		case Uint32:
			_ = data[uint32](t, a)
		case Uint64:
			_ = data[uint64](t, a)
		case Float32:
			_ = data[float32](t, a)
		case Float64:
			_ = data[float64](t, a)
		case Complex64:
			_ = data[complex64](t, a)
		case Complex128:
			_ = data[complex128](t, a)
		case Bool:
			_ = data[bool](t, a)
		}
	}
}
