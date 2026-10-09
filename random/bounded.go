package random

import "math/bits"

// Bounded integers, ports of the random_bounded_*_fill functions of
// numpy/random/src/distributions/distributions.c. Generator uses Lemire's
// multiply-and-reject (masked = false), RandomState the masked rejection of
// NumPy 1.16 (masked = true); both draw 8- and 16-bit values from the halves
// and quarters of a buffered 32-bit output, and the buffer lives only for
// one fill.

// genMask is the smallest all-ones mask covering max (0 for 0).
func genMask(max uint64) uint64 {
	return 1<<bits.Len64(max) - 1
}

// buffered holds the rest of a 32-bit output for 8-, 16- and 1-bit draws.
type buffered struct {
	cnt int
	buf uint32
}

func (b *buffered) uint16(bg BitGenerator) uint16 {
	if b.cnt == 0 {
		b.buf = bg.Uint32()
		b.cnt = 1
	} else {
		b.buf >>= 16
		b.cnt--
	}
	return uint16(b.buf)
}

func (b *buffered) uint8(bg BitGenerator) uint8 {
	if b.cnt == 0 {
		b.buf = bg.Uint32()
		b.cnt = 3
	} else {
		b.buf >>= 8
		b.cnt--
	}
	return uint8(b.buf)
}

func (b *buffered) bool(bg BitGenerator, off, rng bool) bool {
	if !rng {
		return off
	}
	if b.cnt == 0 {
		b.buf = bg.Uint32()
		b.cnt = 31
	} else {
		b.buf >>= 1
		b.cnt--
	}
	return b.buf&1 != 0
}

func lemire64(bg BitGenerator, rng uint64) uint64 {
	excl := rng + 1
	hi, lo := bits.Mul64(bg.Uint64(), excl)
	if lo < excl {
		threshold := (^uint64(0) - rng) % excl
		for lo < threshold {
			hi, lo = bits.Mul64(bg.Uint64(), excl)
		}
	}
	return hi
}

func lemire32(bg BitGenerator, rng uint32) uint32 {
	excl := rng + 1
	m := uint64(bg.Uint32()) * uint64(excl)
	if uint32(m) < excl {
		threshold := (^uint32(0) - rng) % excl
		for uint32(m) < threshold {
			m = uint64(bg.Uint32()) * uint64(excl)
		}
	}
	return uint32(m >> 32)
}

// lemire32PCG is lemire32 on a *PCG64, which the compiler can inline.
func lemire32PCG(bg *PCG64, rng uint32) uint32 {
	excl := rng + 1
	m := uint64(bg.Uint32()) * uint64(excl)
	if uint32(m) < excl {
		threshold := (^uint32(0) - rng) % excl
		for uint32(m) < threshold {
			m = uint64(bg.Uint32()) * uint64(excl)
		}
	}
	return uint32(m >> 32)
}

func lemire16(bg BitGenerator, rng uint16, b *buffered) uint16 {
	excl := rng + 1
	m := uint32(b.uint16(bg)) * uint32(excl)
	if uint16(m) < excl {
		threshold := (^uint16(0) - rng) % excl
		for uint16(m) < threshold {
			m = uint32(b.uint16(bg)) * uint32(excl)
		}
	}
	return uint16(m >> 16)
}

func lemire8(bg BitGenerator, rng uint8, b *buffered) uint8 {
	excl := rng + 1
	m := uint16(b.uint8(bg)) * uint16(excl)
	if uint8(m) < excl {
		threshold := (^uint8(0) - rng) % excl
		for uint8(m) < threshold {
			m = uint16(b.uint8(bg)) * uint16(excl)
		}
	}
	return uint8(m >> 8)
}

// boundedUint64 is numpy's random_bounded_uint64: one value in
// [off, off+rng], wrapping; a range that fits 32 bits draws 32-bit values.
func boundedUint64(bg BitGenerator, off, rng uint64, masked bool) uint64 {
	switch {
	case rng == 0:
		return off
	case rng == 0xFFFFFFFF:
		return off + uint64(bg.Uint32())
	case rng < 0xFFFFFFFF:
		if masked {
			mask := uint32(genMask(rng))
			for {
				if v := bg.Uint32() & mask; uint64(v) <= rng {
					return off + uint64(v)
				}
			}
		}
		return off + uint64(lemire32(bg, uint32(rng)))
	case rng == ^uint64(0):
		return off + bg.Uint64()
	case masked:
		mask := genMask(rng)
		for {
			if v := bg.Uint64() & mask; v <= rng {
				return off + v
			}
		}
	}
	return off + lemire64(bg, rng)
}

// fillUint64 is numpy's random_bounded_uint64_fill: boundedUint64 with
// the choice of method made once, outside the loop.
func fillUint64(bg BitGenerator, off, rng uint64, masked bool, out []uint64) {
	switch {
	case rng == 0 || rng == 0xFFFFFFFF || rng == ^uint64(0) || masked:
		for i := range out {
			out[i] = boundedUint64(bg, off, rng, masked)
		}
	case rng < 0xFFFFFFFF:
		r := uint32(rng)
		if p, ok := bg.(*PCG64); ok {
			for i := range out {
				out[i] = off + uint64(lemire32PCG(p, r))
			}
			return
		}
		for i := range out {
			out[i] = off + uint64(lemire32(bg, r))
		}
	default:
		for i := range out {
			out[i] = off + lemire64(bg, rng)
		}
	}
}

func fillUint32(bg BitGenerator, off, rng uint32, masked bool, out []uint32) {
	mask := uint32(genMask(uint64(rng)))
	for i := range out {
		switch {
		case rng == 0:
			out[i] = off
		case rng == 0xFFFFFFFF:
			out[i] = off + bg.Uint32()
		case masked:
			v := bg.Uint32() & mask
			for v > rng {
				v = bg.Uint32() & mask
			}
			out[i] = off + v
		default:
			out[i] = off + lemire32(bg, rng)
		}
	}
}

func fillUint16(bg BitGenerator, off, rng uint16, masked bool, out []uint16) {
	var b buffered
	mask := uint16(genMask(uint64(rng)))
	for i := range out {
		switch {
		case rng == 0:
			out[i] = off
		case rng == 0xFFFF:
			out[i] = off + b.uint16(bg)
		case masked:
			v := b.uint16(bg) & mask
			for v > rng {
				v = b.uint16(bg) & mask
			}
			out[i] = off + v
		default:
			out[i] = off + lemire16(bg, rng, &b)
		}
	}
}

func fillUint8(bg BitGenerator, off, rng uint8, masked bool, out []uint8) {
	var b buffered
	mask := uint8(genMask(uint64(rng)))
	for i := range out {
		switch {
		case rng == 0:
			out[i] = off
		case rng == 0xFF:
			out[i] = off + b.uint8(bg)
		case masked:
			v := b.uint8(bg) & mask
			for v > rng {
				v = b.uint8(bg) & mask
			}
			out[i] = off + v
		default:
			out[i] = off + lemire8(bg, rng, &b)
		}
	}
}

func fillBool(bg BitGenerator, off, rng bool, out []bool) {
	var b buffered
	for i := range out {
		out[i] = b.bool(bg, off, rng)
	}
}

// interval is numpy's random_interval: a value in [0, max] by masked
// rejection, from 32-bit draws when max fits them. Shuffles use it.
func interval(bg BitGenerator, max uint64) uint64 {
	if max == 0 {
		return 0
	}
	mask := genMask(max)
	if max <= 0xFFFFFFFF {
		for {
			if v := uint64(bg.Uint32()) & mask; v <= max {
				return v
			}
		}
	}
	for {
		if v := bg.Uint64() & mask; v <= max {
			return v
		}
	}
}
