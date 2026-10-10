package random

import (
	"errors"
	"math/big"
	"math/bits"
)

// PCG constants, from numpy/random/src/pcg64/pcg64.h.
var (
	pcgMultiplier = uint128{2549297995355413924, 4865540595714422341}
	// pcgJumpStep is the step of PCG64.jumped: 0x9e3779b97f4a7c15f39cc0605cedc835.
	pcgJumpStep = uint128{0x9e3779b97f4a7c15, 0xf39cc0605cedc835}
)

const pcgCheapMultiplier = 0xda942042e4dd58b5

// ErrNoSeedSequence is returned by Spawn on a bit generator that was not made
// from a SeedSequence (one seeded by legacy seeding or a raw state).
var ErrNoSeedSequence = errors.New("random: the bit generator was not made from a SeedSequence")

// PCG64State is the state of a PCG64 or PCG64DXSM, numpy's state dict: the
// 128-bit LCG state and increment, and the buffered upper half of the last
// 64-bit output.
type PCG64State struct {
	StateHi, StateLo uint64
	IncHi, IncLo     uint64
	HasUint32        bool
	Uinteger         uint32
}

type pcgCore struct {
	state, inc uint128
	half       half32
	seed       *SeedSequence
}

// PCG64 is numpy's default bit generator: a 128-bit LCG with the XSL-RR
// output function.
type PCG64 struct{ pcgCore }

// PCG64DXSM is numpy's PCG64DXSM: PCG64 with the cheap 64-bit multiplier and
// the stronger DXSM output function.
type PCG64DXSM struct{ pcgCore }

// NewPCG64 returns numpy's PCG64(seed): seeded from
// NewSeedSequence(seed...). No seed draws fresh OS entropy.
func NewPCG64(seed ...uint64) *PCG64 { return NewPCG64FromSeedSequence(NewSeedSequence(seed...)) }

// NewPCG64FromSeedSequence returns numpy's PCG64(seed_seq).
func NewPCG64FromSeedSequence(ss *SeedSequence) *PCG64 {
	p := &PCG64{}
	p.seedFrom(ss, p.step)
	return p
}

// NewPCG64DXSM returns numpy's PCG64DXSM(seed).
func NewPCG64DXSM(seed ...uint64) *PCG64DXSM {
	return NewPCG64DXSMFromSeedSequence(NewSeedSequence(seed...))
}

// NewPCG64DXSMFromSeedSequence returns numpy's PCG64DXSM(seed_seq).
func NewPCG64DXSMFromSeedSequence(ss *SeedSequence) *PCG64DXSM {
	// numpy seeds both with pcg64_set_seed, whose srandom steps with the
	// full 128-bit multiplier even for DXSM.
	p := &PCG64DXSM{}
	q := &PCG64{}
	q.seedFrom(ss, q.step)
	p.pcgCore = q.pcgCore
	return p
}

// seedFrom is numpy's pcg64_set_seed after generate_state(4, uint64): the
// words are (state hi, state lo, increment hi, increment lo).
func (c *pcgCore) seedFrom(ss *SeedSequence, step func()) {
	v := ss.GenerateState64(4)
	c.seed = ss
	initState := uint128{v[0], v[1]}
	initSeq := uint128{v[2], v[3]}
	c.state = uint128{}
	c.inc = uint128{initSeq.hi<<1 | initSeq.lo>>63, initSeq.lo<<1 | 1}
	step()
	c.state = c.state.add(initState)
	step()
}

func (p *PCG64) step() { p.state = p.state.mul(pcgMultiplier).add(p.inc) }

func (p *PCG64DXSM) step() {
	hi, lo := bits.Mul64(p.state.lo, pcgCheapMultiplier)
	hi += p.state.hi * pcgCheapMultiplier
	p.state = uint128{hi, lo}.add(p.inc)
}

// Uint64 returns the next 64-bit output.
func (p *PCG64) Uint64() uint64 {
	p.step()
	return bits.RotateLeft64(p.state.hi^p.state.lo, -int(p.state.hi>>58))
}

// Uint64 returns the next 64-bit output: DXSM of the state, then a step.
func (p *PCG64DXSM) Uint64() uint64 {
	hi := p.state.hi
	lo := p.state.lo | 1
	hi ^= hi >> 32
	hi *= pcgCheapMultiplier
	hi ^= hi >> 48
	hi *= lo
	p.step()
	return hi
}

// Uint32 returns the low half of a 64-bit output, or the high half kept from
// the previous call.
func (p *PCG64) Uint32() uint32 {
	if p.half.has {
		p.half.has = false
		return p.half.uint
	}
	return p.half.split(p.Uint64())
}

// Uint32 returns the low half of a 64-bit output, or the high half kept from
// the previous call.
func (p *PCG64DXSM) Uint32() uint32 {
	if p.half.has {
		p.half.has = false
		return p.half.uint
	}
	return p.half.split(p.Uint64())
}

// Float64 returns uint64ToDouble of the next output.
func (p *PCG64) Float64() float64 { return uint64ToDouble(p.Uint64()) }

// Float64 returns uint64ToDouble of the next output.
func (p *PCG64DXSM) Float64() float64 { return uint64ToDouble(p.Uint64()) }

// Raw returns the next 64-bit output.
func (p *PCG64) Raw() uint64 { return p.Uint64() }

// Raw returns the next 64-bit output.
func (p *PCG64DXSM) Raw() uint64 { return p.Uint64() }

// State returns the generator's state.
func (c *pcgCore) State() PCG64State {
	return PCG64State{c.state.hi, c.state.lo, c.inc.hi, c.inc.lo, c.half.has, c.half.uint}
}

// SetState replaces the generator's state.
func (c *pcgCore) SetState(s PCG64State) {
	c.state = uint128{s.StateHi, s.StateLo}
	c.inc = uint128{s.IncHi, s.IncLo}
	c.half = half32{s.HasUint32, s.Uinteger}
}

// SeedSeq returns the SeedSequence the generator was made from.
func (c *pcgCore) SeedSeq() *SeedSequence { return c.seed }

// wrap128 reduces a non-negative or negative integer modulo 2**128, numpy's
// wrap_int(delta, 128).
func wrap128(delta *big.Int) uint128 {
	m := new(big.Int).Lsh(big.NewInt(1), 128)
	d := new(big.Int).Mod(delta, m)
	lo := new(big.Int).And(d, new(big.Int).SetUint64(^uint64(0)))
	return uint128{new(big.Int).Rsh(d, 64).Uint64(), lo.Uint64()}
}

// Advance advances the state as if delta outputs had been drawn, numpy's
// advance(delta); delta is taken modulo 2**128 and may be negative. It drops
// a buffered 32-bit half.
func (p *PCG64) Advance(delta *big.Int) *PCG64 {
	p.state = advanceLCG128(p.state, wrap128(delta), pcgMultiplier, p.inc)
	p.half = half32{}
	return p
}

// Advance advances the state as if delta outputs had been drawn, numpy's
// advance(delta).
func (p *PCG64DXSM) Advance(delta *big.Int) *PCG64DXSM {
	p.state = advanceLCG128(p.state, wrap128(delta), uint128{0, pcgCheapMultiplier}, p.inc)
	p.half = half32{}
	return p
}

func jumpDelta(jumps uint64) *big.Int {
	step := new(big.Int).SetUint64(pcgJumpStep.hi)
	step.Lsh(step, 64).Or(step, new(big.Int).SetUint64(pcgJumpStep.lo))
	return step.Mul(step, new(big.Int).SetUint64(jumps))
}

// Jumped returns a copy advanced by jumps * 0x9e3779b97f4a7c15f39cc0605cedc835
// steps, numpy's jumped(jumps). The copy has no SeedSequence, as in numpy
// (whose copy gets a fresh one that is never used).
func (p *PCG64) Jumped(jumps uint64) *PCG64 {
	q := &PCG64{pcgCore{state: p.state, inc: p.inc, half: p.half}}
	return q.Advance(jumpDelta(jumps))
}

// Jumped returns a copy advanced by jumps * 0x9e3779b97f4a7c15f39cc0605cedc835
// steps, numpy's jumped(jumps).
func (p *PCG64DXSM) Jumped(jumps uint64) *PCG64DXSM {
	q := &PCG64DXSM{pcgCore{state: p.state, inc: p.inc, half: p.half}}
	return q.Advance(jumpDelta(jumps))
}

// Spawn returns n children seeded from SeedSeq().Spawn(n), numpy's spawn.
func (p *PCG64) Spawn(n int) ([]BitGenerator, error) {
	return spawnWith(p.seed, n, func(s *SeedSequence) BitGenerator { return NewPCG64FromSeedSequence(s) })
}

// Spawn returns n children seeded from SeedSeq().Spawn(n), numpy's spawn.
func (p *PCG64DXSM) Spawn(n int) ([]BitGenerator, error) {
	return spawnWith(p.seed, n, func(s *SeedSequence) BitGenerator { return NewPCG64DXSMFromSeedSequence(s) })
}

func spawnWith(ss *SeedSequence, n int, mk func(*SeedSequence) BitGenerator) ([]BitGenerator, error) {
	if ss == nil {
		return nil, ErrNoSeedSequence
	}
	if n < 0 {
		return nil, errors.New("random: n_children must be non-negative")
	}
	kids := ss.Spawn(n)
	out := make([]BitGenerator, n)
	for i, k := range kids {
		out[i] = mk(k)
	}
	return out, nil
}
