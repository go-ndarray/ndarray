package random

import "errors"

const (
	mtN         = 624
	mtM         = 397
	mtMatrixA   = 0x9908b0df
	mtUpperMask = 0x80000000
	mtLowerMask = 0x7fffffff
)

// MT19937State is the state of an MT19937, numpy's state dict: the 624-word
// key and the position in it.
type MT19937State struct {
	Key [mtN]uint32
	Pos int
}

// MT19937 is the Mersenne Twister, numpy's MT19937 and the generator behind
// the legacy RandomState.
type MT19937 struct {
	key  [mtN]uint32
	pos  int
	seed *SeedSequence
}

// NewMT19937 returns numpy's MT19937(seed): seeded from
// NewSeedSequence(seed...). This is not np.random.seed(seed); for that use
// NewMT19937Legacy or RandomState.
func NewMT19937(seed ...uint64) *MT19937 {
	return NewMT19937FromSeedSequence(NewSeedSequence(seed...))
}

// NewMT19937FromSeedSequence returns numpy's MT19937(seed_seq).
func NewMT19937FromSeedSequence(ss *SeedSequence) *MT19937 {
	m := &MT19937{seed: ss}
	m.seedSequence(ss)
	return m
}

// seedSequence is MT19937.__init__: key[0] is 0x80000000, the other words
// come from generate_state(624), and pos is left at 623 (the loop variable
// of numpy's Python loop), so the first output is word 623 untouched.
func (m *MT19937) seedSequence(ss *SeedSequence) {
	v := ss.GenerateState(mtN)
	m.key[0] = 0x80000000
	copy(m.key[1:], v[1:])
	m.pos = mtN - 1
}

// NewMT19937Legacy returns an MT19937 seeded as np.random.seed(seed) seeds
// one: Knuth's initialisation of the reference implementation.
func NewMT19937Legacy(seed uint32) *MT19937 {
	m := &MT19937{}
	m.SeedLegacy(seed)
	return m
}

// SeedLegacy reseeds as np.random.seed(seed) with an integer seed.
func (m *MT19937) SeedLegacy(seed uint32) {
	for pos := range m.key {
		m.key[pos] = seed
		seed = 1812433253*(seed^(seed>>30)) + uint32(pos) + 1
	}
	m.pos = mtN
	m.seed = nil
}

// SeedLegacyArray reseeds as np.random.seed(seed) with an array seed: the
// reference init_by_array.
func (m *MT19937) SeedLegacyArray(key []uint32) error {
	if len(key) == 0 {
		return errors.New("random: Seed must be non-empty")
	}
	mt := &m.key
	mt[0] = 19650218
	for i := 1; i < mtN; i++ {
		mt[i] = 1812433253*(mt[i-1]^(mt[i-1]>>30)) + uint32(i)
	}
	i, j := 1, 0
	for k := max(mtN, len(key)); k > 0; k-- {
		mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525)) + key[j] + uint32(j)
		i++
		j++
		if i >= mtN {
			mt[0] = mt[mtN-1]
			i = 1
		}
		if j >= len(key) {
			j = 0
		}
	}
	for k := mtN - 1; k > 0; k-- {
		mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941)) - uint32(i)
		i++
		if i >= mtN {
			mt[0] = mt[mtN-1]
			i = 1
		}
	}
	mt[0] = 0x80000000
	m.pos = mtN
	m.seed = nil
	return nil
}

func (m *MT19937) gen() {
	k := &m.key
	var i int
	for i = 0; i < mtN-mtM; i++ {
		y := k[i]&mtUpperMask | k[i+1]&mtLowerMask
		k[i] = k[i+mtM] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
	}
	for ; i < mtN-1; i++ {
		y := k[i]&mtUpperMask | k[i+1]&mtLowerMask
		k[i] = k[i+mtM-mtN] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
	}
	y := k[mtN-1]&mtUpperMask | k[0]&mtLowerMask
	k[mtN-1] = k[mtM-1] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
	m.pos = 0
}

// Uint32 returns the next tempered output.
func (m *MT19937) Uint32() uint32 {
	if m.pos == mtN {
		m.gen()
	}
	y := m.key[m.pos]
	m.pos++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// Uint64 returns two outputs, the first in the high half.
func (m *MT19937) Uint64() uint64 {
	hi := uint64(m.Uint32())
	return hi<<32 | uint64(m.Uint32())
}

// Float64 is MT19937's own next_double: 27 bits of one output and 26 of the
// next, (a*2**26 + b) / 2**53.
func (m *MT19937) Float64() float64 {
	a := int32(m.Uint32() >> 5)
	b := int32(m.Uint32() >> 6)
	return (float64(a)*67108864.0 + float64(b)) / 9007199254740992.0
}

// Raw returns the next 32-bit output, widened.
func (m *MT19937) Raw() uint64 { return uint64(m.Uint32()) }

// State returns the generator's state.
func (m *MT19937) State() MT19937State { return MT19937State{m.key, m.pos} }

// SetState replaces the generator's state.
func (m *MT19937) SetState(s MT19937State) {
	m.key = s.Key
	m.pos = s.Pos
}

// SeedSeq returns the SeedSequence the generator was made from, nil after
// legacy seeding.
func (m *MT19937) SeedSeq() *SeedSequence { return m.seed }

// Spawn returns n children seeded from SeedSeq().Spawn(n), numpy's spawn.
func (m *MT19937) Spawn(n int) ([]BitGenerator, error) {
	return spawnWith(m.seed, n, func(s *SeedSequence) BitGenerator { return NewMT19937FromSeedSequence(s) })
}

// Jumped returns a copy advanced by jumps * 2**128 outputs, numpy's
// jumped(jumps), by the reference jump-ahead polynomial.
func (m *MT19937) Jumped(jumps int) *MT19937 {
	c := &MT19937{key: m.key, pos: m.pos}
	for range jumps {
		c.jump()
	}
	return c
}

// genNext advances the state by one word, as the jump code's gen_next.
func (m *MT19937) genNext() {
	k := &m.key
	n := m.pos
	switch {
	case n < mtN-mtM:
		y := k[n]&mtUpperMask | k[n+1]&mtLowerMask
		k[n] = k[n+mtM] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
		m.pos++
	case n < mtN-1:
		y := k[n]&mtUpperMask | k[n+1]&mtLowerMask
		k[n] = k[n+mtM-mtN] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
		m.pos++
	default:
		y := k[mtN-1]&mtUpperMask | k[0]&mtLowerMask
		k[mtN-1] = k[mtM-1] ^ y>>1 ^ (-(y & 1) & mtMatrixA)
		m.pos = 0
	}
}

// addState xors b's state into m's, each read from its own position.
func (m *MT19937) addState(b *MT19937) {
	p1, p2 := m.pos, b.pos
	for i := range mtN {
		m.key[(i+p1)%mtN] ^= b.key[(i+p2)%mtN]
	}
}

func mtCoef(deg int) bool { return mtJumpPoly[deg>>5]&(1<<(deg&31)) != 0 }

// jump is numpy's mt19937_jump_state: Horner evaluation of the jump
// polynomial on the state.
func (m *MT19937) jump() {
	if m.pos >= mtN {
		m.pos = 0
	}
	i := 19937 - 1
	for !mtCoef(i) {
		i--
	}
	t := &MT19937{key: m.key, pos: m.pos}
	t.genNext()
	for i--; i > 0; i-- {
		if mtCoef(i) {
			t.addState(m)
		}
		t.genNext()
	}
	if mtCoef(0) {
		t.addState(m)
	}
	m.key, m.pos = t.key, t.pos
}
