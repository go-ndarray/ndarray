package ndarray

import (
	"bytes"
	"errors"
	"testing"
)

// FuzzUnmarshalBinary feeds arbitrary bytes to UnmarshalBinary, which its
// documentation says may come from anywhere. It must never panic, and:
//
//   - a refusal wraps ErrFormat, so a caller can tell bad input from a bug;
//   - an accepted array re-encodes to at most as many bytes as it was decoded
//     from (the encoding is minimal), so what it holds is bounded by what it
//     was given: a few bytes cannot claim a huge array;
//   - re-encoding is a fixed point: decoding the re-encoding gives the same
//     dtype, shape and bytes.
//
// The seed corpus runs as an ordinary test; `go test -fuzz
// FuzzUnmarshalBinary` explores.
func FuzzUnmarshalBinary(f *testing.F) {
	for _, dt := range allDTypes() {
		for _, shape := range [][]int{{}, {0}, {3}, {2, 3}, {0, maxSize}} {
			a, err := ZerosOf(dt, shape...)
			if err != nil {
				f.Fatal(err)
			}
			b, err := a.MarshalBinary()
			if err != nil {
				f.Fatal(err)
			}
			f.Add(b)
		}
	}
	f.Add([]byte{})
	f.Add([]byte("ND\x01"))
	f.Add([]byte("ND\x01\xff"))
	// A rank and dimensions as large as uvarints can say.
	f.Add([]byte("ND\x01\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"))
	f.Add([]byte("ND\x01\x00\x02\xff\xff\xff\xff\x0f\xff\xff\xff\xff\x0f"))
	// A non-minimal uvarint for a rank of 1, and a bool byte that is not 0 or 1.
	f.Add([]byte("ND\x01\x00\x81\x00\x01\x07"))
	f.Fuzz(func(t *testing.T, data []byte) {
		var a Array
		if err := a.UnmarshalBinary(data); err != nil {
			if !errors.Is(err, ErrFormat) {
				t.Fatalf("refusal does not wrap ErrFormat: %v", err)
			}
			return
		}
		enc, err := a.MarshalBinary()
		if err != nil {
			t.Fatalf("an accepted array does not re-encode: %v", err)
		}
		if len(enc) > len(data) {
			t.Fatalf("%d input bytes decoded to an array of %d encoded bytes (%v %v)",
				len(data), len(enc), a.DType(), a.Shape())
		}
		var b Array
		if err := b.UnmarshalBinary(enc); err != nil {
			t.Fatalf("the re-encoding is refused: %v", err)
		}
		again, err := b.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		if b.DType() != a.DType() || !equalInts(b.Shape(), a.Shape()) || !bytes.Equal(again, enc) {
			t.Fatalf("re-encoding is not a fixed point: %v %v -> %v %v",
				a.DType(), a.Shape(), b.DType(), b.Shape())
		}
	})
}
