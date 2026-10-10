package random

import (
	"math/big"
	"math/bits"
)

// PhiloxState is the state of a Philox, numpy's state dict.
type PhiloxState struct {
	Counter   [4]uint64
	Key       [2]uint64
	Buffer    [4]uint64
	BufferPos int
	HasUint32 bool
	Uinteger  uint32
}

// Philox is numpy's Philox: the counter-based Philox4x64-10 of Salmon et al.,
// four 64-bit outputs per counter value.
type Philox struct {
	ctr    [4]uint64
	key    [2]uint64
	buf    [4]uint64
	bufPos int
	half   half32
	seed   *SeedSequence
}

// NewPhilox returns numpy's Philox(seed): the key from
// NewSeedSequence(seed...).generate_state(2, uint64), the counter zero.
func NewPhilox(seed ...uint64) *Philox {
	return NewPhiloxFromSeedSequence(NewSeedSequence(seed...))
}

// NewPhiloxFromSeedSequence returns numpy's Philox(seed_seq).
func NewPhiloxFromSeedSequence(ss *SeedSequence) *Philox {
	k := ss.GenerateState64(2)
	p := &Philox{key: [2]uint64{k[0], k[1]}, seed: ss}
	p.reset()
	return p
}

// NewPhiloxKey returns numpy's Philox(counter=counter, key=key): counter and
// key as little-endian 64-bit words.
func NewPhiloxKey(counter [4]uint64, key [2]uint64) *Philox {
	p := &Philox{ctr: counter, key: key}
	p.reset()
	return p
}

func (p *Philox) reset() {
	p.half = half32{}
	p.bufPos = 4
	p.buf = [4]uint64{}
}

func philoxRound(ctr [4]uint64, key [2]uint64) [4]uint64 {
	hi0, lo0 := bits.Mul64(0xD2E7470EE14C6C93, ctr[0])
	hi1, lo1 := bits.Mul64(0xCA5A826395121157, ctr[2])
	return [4]uint64{hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0}
}

func philox4x64(ctr [4]uint64, key [2]uint64) [4]uint64 {
	ctr = philoxRound(ctr, key)
	for range 9 {
		key[0] += 0x9E3779B97F4A7C15
		key[1] += 0xBB67AE8584CAA73B
		ctr = philoxRound(ctr, key)
	}
	return ctr
}

// Uint64 returns the next output: the next of the four buffered ones, or the
// first of four new ones after incrementing the counter.
func (p *Philox) Uint64() uint64 {
	if p.bufPos < 4 {
		v := p.buf[p.bufPos]
		p.bufPos++
		return v
	}
	for i := range p.ctr {
		p.ctr[i]++
		if p.ctr[i] != 0 {
			break
		}
	}
	p.buf = philox4x64(p.ctr, p.key)
	p.bufPos = 1
	return p.buf[0]
}

// Uint32 returns the low half of a 64-bit output, or the high half kept from
// the previous call.
func (p *Philox) Uint32() uint32 {
	if p.half.has {
		p.half.has = false
		return p.half.uint
	}
	return p.half.split(p.Uint64())
}

// Float64 returns uint64ToDouble of the next output.
func (p *Philox) Float64() float64 { return uint64ToDouble(p.Uint64()) }

// Raw returns the next 64-bit output.
func (p *Philox) Raw() uint64 { return p.Uint64() }

// State returns the generator's state.
func (p *Philox) State() PhiloxState {
	return PhiloxState{p.ctr, p.key, p.buf, p.bufPos, p.half.has, p.half.uint}
}

// SetState replaces the generator's state.
func (p *Philox) SetState(s PhiloxState) {
	p.ctr, p.key, p.buf, p.bufPos = s.Counter, s.Key, s.Buffer, s.BufferPos
	p.half = half32{s.HasUint32, s.Uinteger}
}

// SeedSeq returns the SeedSequence the generator was made from.
func (p *Philox) SeedSeq() *SeedSequence { return p.seed }

// Spawn returns n children seeded from SeedSeq().Spawn(n), numpy's spawn.
func (p *Philox) Spawn(n int) ([]BitGenerator, error) {
	return spawnWith(p.seed, n, func(s *SeedSequence) BitGenerator { return NewPhiloxFromSeedSequence(s) })
}

// Advance adds delta (taken modulo 2**256, may be negative) to the counter,
// numpy's advance(delta), and drops the buffered outputs.
func (p *Philox) Advance(delta *big.Int) *Philox {
	m := new(big.Int).Lsh(big.NewInt(1), 256)
	d := new(big.Int).Mod(delta, m)
	var step [4]uint64
	mask := new(big.Int).SetUint64(^uint64(0))
	for i := range step {
		step[i] = new(big.Int).And(d, mask).Uint64()
		d.Rsh(d, 64)
	}
	carry := false
	for i := range p.ctr {
		if carry {
			p.ctr[i]++
			carry = p.ctr[i] == 0
		}
		orig := p.ctr[i]
		p.ctr[i] += step[i]
		if p.ctr[i] < orig && !carry {
			carry = true
		}
	}
	p.reset()
	return p
}

// Jumped returns a copy advanced by jumps * 2**128 counter steps, numpy's
// jumped(jumps).
func (p *Philox) Jumped(jumps uint64) *Philox {
	q := &Philox{ctr: p.ctr, key: p.key, buf: p.buf, bufPos: p.bufPos, half: p.half}
	d := new(big.Int).Lsh(new(big.Int).SetUint64(jumps), 128)
	return q.Advance(d)
}

// SFC64State is the state of an SFC64, numpy's state dict.
type SFC64State struct {
	S         [4]uint64
	HasUint32 bool
	Uinteger  uint32
}

// SFC64 is numpy's SFC64, Chris Doty-Humphrey's Small Fast Chaotic generator.
type SFC64 struct {
	s    [4]uint64
	half half32
	seed *SeedSequence
}

// NewSFC64 returns numpy's SFC64(seed).
func NewSFC64(seed ...uint64) *SFC64 { return NewSFC64FromSeedSequence(NewSeedSequence(seed...)) }

// NewSFC64FromSeedSequence returns numpy's SFC64(seed_seq): three words from
// generate_state(3, uint64), a counter of 1, then twelve discarded outputs.
func NewSFC64FromSeedSequence(ss *SeedSequence) *SFC64 {
	v := ss.GenerateState64(3)
	g := &SFC64{s: [4]uint64{v[0], v[1], v[2], 1}, seed: ss}
	for range 12 {
		g.Uint64()
	}
	return g
}

// Uint64 returns the next output.
func (g *SFC64) Uint64() uint64 {
	s := &g.s
	tmp := s[0] + s[1] + s[3]
	s[3]++
	s[0] = s[1] ^ (s[1] >> 11)
	s[1] = s[2] + (s[2] << 3)
	s[2] = bits.RotateLeft64(s[2], 24) + tmp
	return tmp
}

// Uint32 returns the low half of a 64-bit output, or the high half kept from
// the previous call.
func (g *SFC64) Uint32() uint32 {
	if g.half.has {
		g.half.has = false
		return g.half.uint
	}
	return g.half.split(g.Uint64())
}

// Float64 returns uint64ToDouble of the next output.
func (g *SFC64) Float64() float64 { return uint64ToDouble(g.Uint64()) }

// Raw returns the next 64-bit output.
func (g *SFC64) Raw() uint64 { return g.Uint64() }

// State returns the generator's state.
func (g *SFC64) State() SFC64State { return SFC64State{g.s, g.half.has, g.half.uint} }

// SetState replaces the generator's state.
func (g *SFC64) SetState(s SFC64State) { g.s, g.half = s.S, half32{s.HasUint32, s.Uinteger} }

// SeedSeq returns the SeedSequence the generator was made from.
func (g *SFC64) SeedSeq() *SeedSequence { return g.seed }

// Spawn returns n children seeded from SeedSeq().Spawn(n), numpy's spawn.
func (g *SFC64) Spawn(n int) ([]BitGenerator, error) {
	return spawnWith(g.seed, n, func(s *SeedSequence) BitGenerator { return NewSFC64FromSeedSequence(s) })
}
