package ndarray

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"strings"
	"testing"
)

func TestMarshalRoundTrip(t *testing.T) {
	for _, dt := range allDTypes() {
		a := must(Arange(0, 12, 1)).AsType(dt)
		a = must(a.Reshape(3, 4))
		if dt == Bool {
			a = must(FromSlice([]bool{true, false, true, true, false, false, true, false, true, true, true, false}, 3, 4))
		}
		if dt.IsComplex() {
			a = must(a.Add(Scalar(-2.5i)))
		}
		// A strided view is encoded as the array it shows.
		v := must(a.Slice(All(), Step(2)))
		for _, x := range []*Array{a, v, must(ZerosOf(dt)), must(ZerosOf(dt, 0, 5))} {
			b, err := x.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			var y Array
			if err := y.UnmarshalBinary(b); err != nil {
				t.Fatalf("%v %v: %v", dt, x.Shape(), err)
			}
			if y.DType() != dt || y.String() != x.String() {
				t.Errorf("%v round trip: %v, want %v", dt, y.String(), x.String())
			}
		}
	}
}

// TestMarshalLayout pins the bytes, so that every architecture (s390x is
// big-endian) writes the same ones.
func TestMarshalLayout(t *testing.T) {
	a := must(FromSlice([]int16{1, -2}, 2))
	b, _ := a.MarshalBinary()
	want := []byte{'N', 'D', 1, byte(Int16), 1, 2, 1, 0, 0xfe, 0xff}
	if !bytes.Equal(b, want) {
		t.Errorf("bytes = %v, want %v", b, want)
	}
}

func TestUnmarshalRejects(t *testing.T) {
	good, _ := must(FromSlice([]float64{1, 2, 3}, 3)).MarshalBinary()
	head := func(dt DType, dims ...uint64) []byte {
		b := []byte{'N', 'D', 1, byte(dt)}
		b = binary.AppendUvarint(b, uint64(len(dims)))
		for _, d := range dims {
			b = binary.AppendUvarint(b, d)
		}
		return b
	}
	cases := map[string][]byte{
		"empty":       nil,
		"magic":       append([]byte("XD"), good[2:]...),
		"version":     append([]byte{'N', 'D', 2}, good[3:]...),
		"dtype":       head(DType(99)),
		"no rank":     []byte{'N', 'D', 1, 0},
		"rank":        head(Float64, make([]uint64, 65)...),
		"cut dim":     append(head(Float64), 1),
		"huge dim":    head(Float64, 1<<62),
		"overflow":    head(Float64, 1<<40, 1<<40),
		"short data":  append(head(Float64, 3), 1, 2, 3),
		"ragged data": append(head(Complex128, 1), make([]byte, 17)...),
		"long data":   append(head(Float64, 1), make([]byte, 16)...),
		"lying shape": head(Complex128, 1<<58),
	}
	for name, b := range cases {
		var a Array
		if err := a.UnmarshalBinary(b); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func TestGobKeepsArrays(t *testing.T) {
	type holder struct{ A *Array }
	h := holder{A: must(FromSlice([]complex64{1 + 2i}, 1))}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(h); err != nil {
		t.Fatal(err)
	}
	var h2 holder
	if err := gob.NewDecoder(&buf).Decode(&h2); err != nil || Item[complex64](h2.A, 0) != 1+2i {
		t.Errorf("gob: %v %v", h2.A, err)
	}
}

func TestRepr(t *testing.T) {
	big := must(Arange(0, 2000, 1))
	cases := []struct {
		a    *Array
		want string
	}{
		{must(FromSlice([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 4)),
			"array([[ 0,  1,  2,  3],\n       [ 4,  5,  6,  7],\n       [ 8,  9, 10, 11]])"},
		{must(FromSlice([]float32{1.5, 2, 3}, 3)), "array([1.5,  2.,  3.], dtype=float32)"},
		{must(FromSlice([]complex128{1 + 2i, complex(0, -1)}, 2)), "array([1.+2.j, 0.-1.j])"},
		{must(FromSlice([]complex64{complex(0, float32(negZero()))}, 1)), "array([0.-0.j], dtype=complex64)"},
		{must(FromSlice([]bool{true, false}, 2)), "array([ True, False])"},
		{must(FromSlice([]uint8{7}, []int{}...)), "array(7, dtype=uint8)"},
		{must(FromSlice([]float64{1e20, 0.5}, 2)), "array([1e+20,   0.5])"},
		{big, "array([   0.,    1.,    2., ...,"},
		{must(big.Reshape(40, 50)), "       ...,\n"},
		{must(big.Reshape(40, 50)), "shape=(40, 50))"},
		{big, "shape=(2000,))"},
		{must(must(Arange(0, 24, 1)).Reshape(2, 3, 4)), "11.]],\n\n       [[12."},
		{must(Arange(0, 40, 1)), "12.,\n       13."},
	}
	for _, c := range cases {
		if got := c.a.Repr(); !strings.Contains(got, c.want) {
			t.Errorf("Repr =\n%s\nwant it to contain\n%s", got, c.want)
		}
	}
}

func negZero() float64 { z := 0.0; return -z }

func TestDisplay(t *testing.T) {
	m := must(FromSlice([]int64{1, 2, 3, 4}, 2, 2))
	d := m.Display()
	if string(d["text/plain"]) != m.Repr() {
		t.Error("text/plain is not the Repr")
	}
	h := string(d["text/html"])
	for _, want := range []string{`<caption>(2, 2) int64</caption>`, "<th>1</th>", "<td>4</td>"} {
		if !strings.Contains(h, want) {
			t.Errorf("html lacks %q:\n%s", want, h)
		}
	}
	if h := string(must(FromSlice([]bool{true}, 1)).Display()["text/html"]); !strings.Contains(h, "<td>True</td>") || !strings.Contains(h, "(1,) bool") || strings.Contains(h, "<thead>") {
		t.Errorf("1-D html:\n%s", h)
	}
	big := must(must(Arange(0, 4000, 1)).Reshape(40, 100))
	if h := string(big.Display()["text/html"]); !strings.Contains(h, "<th>⋯</th>") || !strings.Contains(h, "<td>⋯</td>") || !strings.Contains(h, "<td>3999.</td>") {
		t.Errorf("summarised html")
	}
	if _, ok := must(Zeros(2, 2, 2)).Display()["text/html"]; ok {
		t.Error("3-D html")
	}
	if shapeString([]int{}) != "()" {
		t.Error("0-d shape")
	}
}
