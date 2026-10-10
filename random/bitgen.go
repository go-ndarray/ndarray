package random

import "math/bits"

// BitGenerator is a source of raw random bits, numpy's BitGenerator. Every
// distribution draws from one of these three streams, and which one it draws
// from is part of numpy's output: a 32-bit draw from PCG64 uses half of a
// 64-bit output and keeps the other half for the next 32-bit draw, while
// MT19937 makes a 64-bit value from two 32-bit outputs.
//
// The implementations in this package are not safe for concurrent use.
type BitGenerator interface {
	// Uint64 is numpy's next_uint64.
	Uint64() uint64
	// Uint32 is numpy's next_uint32.
	Uint32() uint32
	// Float64 is numpy's next_double: a float64 in [0, 1) with 53 random
	// bits.
	Float64() float64
	// Raw is numpy's next_raw, the stream random_raw returns.
	Raw() uint64
}

// Spawner is a BitGenerator made from a SeedSequence, which can make
// independent children, numpy's BitGenerator.spawn.
type Spawner interface {
	BitGenerator
	// SeedSeq returns the SeedSequence the generator was made from, nil
	// when it was not made from one.
	SeedSeq() *SeedSequence
	// Spawn returns n children made from SeedSeq().Spawn(n).
	Spawn(n int) ([]BitGenerator, error)
}

// RandomRaw returns n outputs of bg's raw stream, numpy's random_raw(n).
func RandomRaw(bg BitGenerator, n int) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		out[i] = bg.Raw()
	}
	return out
}

// uint64ToDouble is numpy's uint64_to_double: the top 53 bits scaled to
// [0, 1).
func uint64ToDouble(r uint64) float64 {
	return float64(r>>11) * (1.0 / 9007199254740992.0)
}

// half32 holds the upper half of a 64-bit output for the next Uint32, as
// numpy's has_uint32/uinteger pair does.
type half32 struct {
	has  bool
	uint uint32
}

// split keeps the upper half of v for the next 32-bit draw and returns the
// lower half.
func (h *half32) split(v uint64) uint32 {
	h.has = true
	h.uint = uint32(v >> 32)
	return uint32(v)
}

// uint128 is a 128-bit unsigned integer.
type uint128 struct{ hi, lo uint64 }

func (a uint128) add(b uint128) uint128 {
	lo, c := bits.Add64(a.lo, b.lo, 0)
	hi, _ := bits.Add64(a.hi, b.hi, c)
	return uint128{hi, lo}
}

func (a uint128) mul(b uint128) uint128 {
	hi, lo := bits.Mul64(a.lo, b.lo)
	hi += a.hi*b.lo + a.lo*b.hi
	return uint128{hi, lo}
}

func (a uint128) isZero() bool { return a.hi == 0 && a.lo == 0 }

func (a uint128) shr1() uint128 { return uint128{a.hi >> 1, a.lo>>1 | a.hi<<63} }

// advanceLCG128 is numpy's pcg_advance_lcg_128: the state of the LCG
// state*mult+plus after delta steps, in O(log delta).
func advanceLCG128(state, delta, mult, plus uint128) uint128 {
	accMult := uint128{0, 1}
	accPlus := uint128{}
	for !delta.isZero() {
		if delta.lo&1 != 0 {
			accMult = accMult.mul(mult)
			accPlus = accPlus.mul(mult).add(plus)
		}
		plus = mult.add(uint128{0, 1}).mul(plus)
		mult = mult.mul(mult)
		delta = delta.shr1()
	}
	return accMult.mul(state).add(accPlus)
}
