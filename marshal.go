package ndarray

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// The binary form of an array, as MarshalBinary writes it:
//
//	"ND" 0x01            magic and version
//	dtype                one byte, the DType
//	ndim                 uvarint
//	shape[0..ndim)       uvarints
//	elements             row-major, little-endian, DType.Size bytes each
//	                     (a bool is one byte, 0 or 1; a complex number is its
//	                     real part, then its imaginary part)
//
// The same bytes come out on every architecture.

var marshalMagic = [3]byte{'N', 'D', 1}

// ErrFormat is returned by UnmarshalBinary for bytes that are not an array.
var ErrFormat = errors.New("ndarray: not an encoded array")

// MarshalBinary encodes the array: its dtype, its shape and its elements in
// row-major order. A view is encoded as the array it shows. It implements
// encoding.BinaryMarshaler, so an array can be kept by gob and by a
// notebook kernel.
func (a *Array) MarshalBinary() ([]byte, error) {
	b := make([]byte, 0, 16+a.Size()*a.dtype.Size())
	b = append(b, marshalMagic[:]...)
	b = append(b, byte(a.dtype))
	b = binary.AppendUvarint(b, uint64(len(a.shape)))
	for _, d := range a.shape {
		b = binary.AppendUvarint(b, uint64(d))
	}
	return binary.Append(b, binary.LittleEndian, a.contiguousStore())
}

// UnmarshalBinary decodes what MarshalBinary wrote into a, replacing its
// contents. The bytes may come from anywhere: a shape the length of the data
// does not account for is an error before anything is allocated.
func (a *Array) UnmarshalBinary(b []byte) error {
	if len(b) < 4 || [3]byte(b[:3]) != marshalMagic {
		return ErrFormat
	}
	dt := DType(b[3])
	if int(dt) >= numDTypes {
		return fmt.Errorf("%w: dtype %d", ErrFormat, b[3])
	}
	b = b[4:]
	ndim, n := binary.Uvarint(b)
	if n <= 0 || ndim > 64 {
		return fmt.Errorf("%w: bad rank", ErrFormat)
	}
	b = b[n:]
	shape := make([]int, ndim)
	for i := range shape {
		d, n := binary.Uvarint(b)
		if n <= 0 || d > maxSize {
			return fmt.Errorf("%w: bad dimension", ErrFormat)
		}
		shape[i], b = int(d), b[n:]
	}
	if err := validateShape(shape); err != nil {
		return fmt.Errorf("%w: %v", ErrFormat, err)
	}
	// Divide rather than multiply: size*16 can overflow for a complex128
	// shape that validateShape (which counts 8 bytes an element) allows.
	if size := prod(shape); len(b)%dt.Size() != 0 || len(b)/dt.Size() != size {
		return fmt.Errorf("%w: %d bytes for %d elements of %v", ErrFormat, len(b), size, dt)
	}
	s := makeStore(dt, prod(shape))
	// The length was checked above, so Decode cannot run short.
	_, _ = binary.Decode(b, binary.LittleEndian, s)
	*a = *fromStore(s, shape)
	return nil
}
