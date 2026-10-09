package ndarray

import (
	_ "embed"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// promotion.json is generated from NumPy by testdata/promotion.py. It is
// embedded so the emulated CI arches, which run the test binary from another
// directory, can read it.
//
//go:embed testdata/promotion.json
var promotionJSON []byte

type promotionTable struct {
	NumPy  string                       `json:"numpy"`
	Pair   map[string]string            `json:"pair"`
	Weak   map[string]map[string]string `json:"weak"`
	Unary  map[string]map[string]string `json:"unary"`
	Reduce map[string]map[string]string `json:"reduce"`
}

func loadPromotion(t *testing.T) promotionTable {
	t.Helper()
	var p promotionTable
	if err := json.Unmarshal(promotionJSON, &p); err != nil {
		t.Fatal(err)
	}
	if len(p.Pair) != numDTypes*numDTypes {
		t.Fatalf("fixture has %d pairs, want %d", len(p.Pair), numDTypes*numDTypes)
	}
	return p
}

func allDTypes() []DType {
	out := make([]DType, numDTypes)
	for i := range out {
		out[i] = DType(i)
	}
	return out
}

// noFloat16 maps NumPy's float16 results to float32, the documented
// difference.
func noFloat16(s string) string {
	if s == "float16" {
		return "float32"
	}
	return s
}

func TestDTypeNames(t *testing.T) {
	for _, d := range allDTypes() {
		got, err := ParseDType(d.String())
		if err != nil || got != d {
			t.Errorf("ParseDType(%q) = %v, %v", d.String(), got, err)
		}
	}
	if _, err := ParseDType("float16"); !errors.Is(err, ErrDType) {
		t.Errorf("ParseDType(float16) err = %v", err)
	}
	if s := DType(200).String(); s != "DType(200)" {
		t.Errorf("String = %q", s)
	}
}

func TestDTypeKinds(t *testing.T) {
	sizes := map[DType]int{Bool: 1, Int8: 1, Uint8: 1, Int16: 2, Uint16: 2, Int32: 4, Uint32: 4,
		Float32: 4, Int64: 8, Uint64: 8, Float64: 8, Complex64: 8, Complex128: 16}
	for d, want := range sizes {
		if d.Size() != want {
			t.Errorf("%v.Size() = %d, want %d", d, d.Size(), want)
		}
	}
	if !Bool.IsBool() || Int8.IsBool() || !Int8.IsSigned() || Uint8.IsSigned() ||
		!Uint64.IsInteger() || Float32.IsInteger() || !Float32.IsFloat() || !Complex64.IsComplex() {
		t.Error("kind predicates disagree")
	}
}

// TestResultTypeMatchesNumPy checks ResultType, and the dtype the arithmetic
// actually produces, against every cell of numpy.result_type.
func TestResultTypeMatchesNumPy(t *testing.T) {
	p := loadPromotion(t)
	for _, a := range allDTypes() {
		for _, b := range allDTypes() {
			want := p.Pair[a.String()+","+b.String()]
			if got := ResultType(a, b).String(); got != want {
				t.Errorf("ResultType(%v, %v) = %v, NumPy %s", a, b, got, want)
			}
			x, _ := OnesOf(a, 2)
			y, _ := OnesOf(b, 2)
			op := x.Add
			if a == Bool && b == Bool {
				op = x.Mul // bool - bool is an error; bool * bool is bool
			}
			r, err := op(y)
			if err != nil || r.DType().String() != want {
				t.Errorf("%v + %v gives %v (%v), NumPy %s", a, b, r.DType(), err, want)
			}
		}
	}
}

// TestWeakScalarsMatchNumPy checks the NEP 50 results of an array with a Go
// int, float and complex scalar, and of true division.
func TestWeakScalarsMatchNumPy(t *testing.T) {
	p := loadPromotion(t)
	for _, d := range allDTypes() {
		w := p.Weak[d.String()]
		x, _ := OnesOf(d, 2)
		for kind, s := range map[string]*Array{"int": Scalar(1), "float": Scalar(1.5), "complex": Scalar(1i)} {
			r, err := x.Add(s)
			if err != nil || r.DType().String() != w[kind] {
				t.Errorf("%v + %s scalar = %v (%v), NumPy %s", d, kind, r.DType(), err, w[kind])
			}
			r, err = s.Add(x)
			if err != nil || r.DType().String() != w[kind] {
				t.Errorf("%s scalar + %v = %v (%v), NumPy %s", kind, d, r.DType(), err, w[kind])
			}
		}
		r, err := x.Div(x)
		if err != nil || r.DType().String() != w["truediv"] {
			t.Errorf("%v / %v = %v (%v), NumPy %s", d, d, r.DType(), err, w["truediv"])
		}
	}
	if r, _ := Scalar(1).Add(Scalar(2.5)); r.DType() != Float64 {
		t.Errorf("int scalar + float scalar = %v", r.DType())
	}
	if r, _ := Scalar(int8(1)).Add(Scalar(uint(1))); r.DType() != Int64 {
		t.Errorf("int scalar + uint scalar = %v", r.DType())
	}
	if r, _ := Scalar(true).Add(Scalar(float32(1))); r.DType() != Float64 {
		t.Errorf("bool + float scalar = %v", r.DType())
	}
	if r, _ := Scalar(complex64(1)).Add(Scalar(1)); r.DType() != Complex128 {
		t.Errorf("complex scalar + int scalar = %v", r.DType())
	}
}

func TestScalarPanicsOnOtherTypes(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Scalar(string) did not panic")
		}
	}()
	Scalar("1")
}

// TestUnaryDTypesMatchNumPy checks the dtype of every unary ufunc result
// against NumPy, with float16 read as float32 and NumPy's TypeErrors as
// panics.
func TestUnaryDTypesMatchNumPy(t *testing.T) {
	p := loadPromotion(t)
	ops := map[string]func(*Array) *Array{
		"sqrt": (*Array).Sqrt, "exp": (*Array).Exp, "sin": (*Array).Sin, "abs": (*Array).Abs,
		"negative": (*Array).Neg, "floor": (*Array).Floor, "square": (*Array).Square,
	}
	for _, d := range allDTypes() {
		for name, op := range ops {
			want := noFloat16(p.Unary[d.String()][name])
			x, _ := OnesOf(d, 3)
			got, panicked := func() (s string, panicked bool) {
				defer func() {
					if recover() != nil {
						panicked = true
					}
				}()
				return op(x).DType().String(), false
			}()
			if strings.HasPrefix(want, "ERR") {
				if !panicked {
					t.Errorf("%s(%v) = %s, NumPy raises", name, d, got)
				}
				continue
			}
			if panicked || got != want {
				t.Errorf("%s(%v) = %s (panic %v), NumPy %s", name, d, got, panicked, want)
			}
		}
	}
}

// TestReductionDTypesMatchNumPy checks the dtype of the axis reductions.
func TestReductionDTypesMatchNumPy(t *testing.T) {
	p := loadPromotion(t)
	ops := map[string]func(*Array) (*Array, error){
		"sum":    func(a *Array) (*Array, error) { return a.SumAxis(0, false) },
		"prod":   func(a *Array) (*Array, error) { return a.ProdAxis(0, false) },
		"mean":   func(a *Array) (*Array, error) { return a.MeanAxis(0, false) },
		"max":    func(a *Array) (*Array, error) { return a.MaxAxis(0, false) },
		"cumsum": func(a *Array) (*Array, error) { return a.CumSum(0) },
		"argmax": func(a *Array) (*Array, error) { return a.ArgMaxAxis(0, false) },
	}
	for _, d := range allDTypes() {
		for name, op := range ops {
			want := p.Reduce[d.String()][name]
			x, _ := OnesOf(d, 3)
			r, err := op(x)
			if err != nil || r.DType().String() != want {
				t.Errorf("%s(%v) = %v (%v), NumPy %s", name, d, r.DType(), err, want)
			}
		}
	}
}
