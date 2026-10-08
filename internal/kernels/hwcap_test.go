package kernels

import (
	"encoding/binary"
	"testing"
)

func auxvOf(pairs ...uint64) []byte {
	b := make([]byte, 8*len(pairs))
	for i, p := range pairs {
		binary.NativeEndian.PutUint64(b[8*i:], p)
	}
	return b
}

func TestHWCAPFrom(t *testing.T) {
	for _, c := range []struct {
		name string
		auxv []byte
		want uint64
	}{
		{"found after others", auxvOf(6, 4096, atHWCAP, 0x3f, 0, 0), 0x3f},
		{"absent", auxvOf(6, 4096, 0, 0), 0},
		{"after the terminator", auxvOf(0, 0, atHWCAP, 1), 0},
		{"empty", nil, 0},
		{"short tail", append(auxvOf(6, 1), 1, 2, 3), 0},
	} {
		if got := hwcapFrom(c.auxv); got != c.want {
			t.Errorf("%s: %#x, want %#x", c.name, got, c.want)
		}
	}
}
