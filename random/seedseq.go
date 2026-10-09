package random

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math/big"
)

// SeedSequence constants, from numpy/random/bit_generator.pyx (after
// Melissa O'Neill's seed_seq_fe).
const (
	defaultPoolSize = 4
	initA           = 0x43b0d7e5
	multA           = 0x931e8875
	initB           = 0x8b51f9dd
	multB           = 0x58f38ded
	mixMultL        = 0xca01f9dd
	mixMultR        = 0x4973f715
	xshift          = 16
)

// SeedSequence mixes user entropy into a well-distributed pool and derives
// bit-generator states from it, exactly as numpy.random.SeedSequence: the
// same entropy, spawn key and pool size give the same pool, the same
// GenerateState words and the same children.
//
// A SeedSequence is not safe for concurrent use (Spawn mutates it).
type SeedSequence struct {
	entropy          []uint32 // the entropy as uint32 words, lowest first
	spawnKey         []uint32 // the spawn key as uint32 words
	spawnKeyInts     []uint64 // the spawn key as given
	poolSize         int
	nChildrenSpawned uint64
	pool             []uint32
}

// NewSeedSequence returns the SeedSequence of numpy's SeedSequence(entropy).
// One value is numpy's integer entropy (SeedSequence(42)); several are its
// sequence form (SeedSequence([1, 2, 3])), where each value is split into
// uint32 words, lowest first. With no value the entropy is 128 fresh bits
// from the operating system, as SeedSequence() draws them.
//
// For an integer wider than 64 bits use NewSeedSequenceBig. Python's entropy
// 2**64 + 5, for instance, is the sequence [5, 0, 1] of words and not
// expressible as uint64s: [5, 2**32] would be [5, 0, 1] as well — numpy
// itself cannot tell those apart, and neither can this.
func NewSeedSequence(entropy ...uint64) *SeedSequence {
	return newSeedSequence(entropyWords(entropy, defaultPoolSize), nil, defaultPoolSize, 0)
}

// NewSeedSequenceBig returns the SeedSequence of numpy's
// SeedSequence(entropy) for an arbitrary non-negative integer.
func NewSeedSequenceBig(entropy *big.Int) (*SeedSequence, error) {
	words, err := bigWords(entropy)
	if err != nil {
		return nil, err
	}
	return NewSeedSequenceWith(SeedSequenceOptions{Words: words})
}

// SeedSequenceOptions are the arguments of numpy's SeedSequence constructor.
type SeedSequenceOptions struct {
	// Entropy is numpy's entropy as integers (see NewSeedSequence).
	Entropy []uint64
	// Words, when not nil, is the entropy as uint32 words, lowest first,
	// and Entropy must be empty. It expresses any Python integer or list.
	Words []uint32
	// SpawnKey is numpy's spawn_key.
	SpawnKey []uint64
	// PoolSize is numpy's pool_size; zero means the default, 4.
	PoolSize int
	// NChildrenSpawned is numpy's n_children_spawned.
	NChildrenSpawned uint64
}

// NewSeedSequenceWith returns the SeedSequence of numpy's
// SeedSequence(entropy, spawn_key=..., pool_size=..., n_children_spawned=...).
// Entropy and Words both empty draw fresh entropy from the OS.
func NewSeedSequenceWith(o SeedSequenceOptions) (*SeedSequence, error) {
	pool := o.PoolSize
	if pool == 0 {
		pool = defaultPoolSize
	}
	if pool < defaultPoolSize {
		return nil, fmt.Errorf("random: the size of the entropy pool should be at least %d", defaultPoolSize)
	}
	if len(o.Words) > 0 && len(o.Entropy) > 0 {
		return nil, fmt.Errorf("random: Entropy and Words are exclusive")
	}
	var words []uint32
	if o.Words != nil {
		words = make([]uint32, len(o.Words))
		copy(words, o.Words)
	} else {
		words = entropyWords(o.Entropy, pool)
	}
	return newSeedSequence(words, o.SpawnKey, pool, o.NChildrenSpawned), nil
}

// entropyWords is the uint32 words of numpy's integer (one value) or
// sequence entropy, or fresh OS entropy for none.
func entropyWords(entropy []uint64, pool int) []uint32 {
	switch len(entropy) {
	case 0:
		return osEntropy(pool)
	case 1:
		return intWords(entropy[0])
	}
	return sequenceWords(entropy)
}

func newSeedSequence(words []uint32, spawnKey []uint64, pool int, spawned uint64) *SeedSequence {
	ss := &SeedSequence{
		entropy:          words,
		spawnKey:         sequenceWords(spawnKey),
		spawnKeyInts:     append([]uint64(nil), spawnKey...),
		poolSize:         pool,
		nChildrenSpawned: spawned,
		pool:             make([]uint32, pool),
	}
	ss.mixEntropy(ss.assembledEntropy())
	return ss
}

// readOSEntropy fills b from the operating system; a variable so a test can
// observe the call.
var readOSEntropy = rand.Read

// osEntropy draws pool*32 bits, as numpy's randbits(pool_size * 32), and
// returns them as the words of that integer (leading zero words dropped).
func osEntropy(pool int) []uint32 {
	b := make([]byte, 4*pool)
	_, _ = readOSEntropy(b) // crypto/rand.Read never fails (Go 1.24+)
	w := make([]uint32, pool)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return trimWords(w)
}

// intWords splits a non-negative integer into uint32 words, lowest first;
// zero is the single word 0 (numpy's _int_to_uint32_array).
func intWords(v uint64) []uint32 {
	if v == 0 {
		return []uint32{0}
	}
	var w []uint32
	for v > 0 {
		w = append(w, uint32(v))
		v >>= 32
	}
	return w
}

// sequenceWords concatenates the words of each integer, as numpy's
// _coerce_to_uint32_array does for a sequence.
func sequenceWords(vs []uint64) []uint32 {
	var w []uint32
	for _, v := range vs {
		w = append(w, intWords(v)...)
	}
	return w
}

// trimWords drops high zero words, keeping at least one: the words of the
// integer the slice spells.
func trimWords(w []uint32) []uint32 {
	n := len(w)
	for n > 1 && w[n-1] == 0 {
		n--
	}
	return w[:n]
}

// bigWords returns the uint32 words of a non-negative big integer.
func bigWords(v *big.Int) ([]uint32, error) {
	if v.Sign() < 0 {
		return nil, fmt.Errorf("random: expected non-negative integer")
	}
	b := v.Bytes() // big-endian
	w := make([]uint32, (len(b)+3)/4)
	for i := range b {
		pos := len(b) - 1 - i // little-endian byte index
		w[pos/4] |= uint32(b[i]) << (8 * (pos % 4))
	}
	if len(w) == 0 {
		w = []uint32{0}
	}
	return w, nil
}

func hashmix(value uint32, hashConst *uint32) uint32 {
	value ^= *hashConst
	*hashConst *= multA
	value *= *hashConst
	value ^= value >> xshift
	return value
}

func mix(x, y uint32) uint32 {
	r := mixMultL*x - mixMultR*y
	r ^= r >> xshift
	return r
}

// assembledEntropy is numpy's get_assembled_entropy: the run entropy, padded
// with zeros to the pool size when there is a spawn key, then the spawn key.
func (s *SeedSequence) assembledEntropy() []uint32 {
	run := append([]uint32(nil), s.entropy...)
	if len(s.spawnKey) > 0 && len(run) < s.poolSize {
		run = append(run, make([]uint32, s.poolSize-len(run))...)
	}
	return append(run, s.spawnKey...)
}

func (s *SeedSequence) mixEntropy(e []uint32) {
	h := uint32(initA)
	m := s.pool
	for i := range m {
		if i < len(e) {
			m[i] = hashmix(e[i], &h)
		} else {
			m[i] = hashmix(0, &h)
		}
	}
	for src := range m {
		for dst := range m {
			if src != dst {
				m[dst] = mix(m[dst], hashmix(m[src], &h))
			}
		}
	}
	for src := len(m); src < len(e); src++ {
		for dst := range m {
			m[dst] = mix(m[dst], hashmix(e[src], &h))
		}
	}
}

// Entropy returns the entropy as uint32 words, lowest first.
func (s *SeedSequence) Entropy() []uint32 { return append([]uint32(nil), s.entropy...) }

// EntropyInt returns the entropy as the integer numpy shows for an integer
// entropy (the words read as one little-endian number).
func (s *SeedSequence) EntropyInt() *big.Int {
	v := new(big.Int)
	for i := len(s.entropy) - 1; i >= 0; i-- {
		v.Lsh(v, 32)
		v.Or(v, big.NewInt(int64(s.entropy[i])))
	}
	return v
}

// SpawnKey returns the spawn key.
func (s *SeedSequence) SpawnKey() []uint64 { return append([]uint64(nil), s.spawnKeyInts...) }

// PoolSize returns the pool size in uint32 words.
func (s *SeedSequence) PoolSize() int { return s.poolSize }

// NChildrenSpawned returns how many children Spawn has made so far.
func (s *SeedSequence) NChildrenSpawned() uint64 { return s.nChildrenSpawned }

// Pool returns a copy of the mixed entropy pool.
func (s *SeedSequence) Pool() []uint32 { return append([]uint32(nil), s.pool...) }

// GenerateState returns n uint32 words of state, numpy's
// generate_state(n, np.uint32).
func (s *SeedSequence) GenerateState(n int) []uint32 {
	h := uint32(initB)
	out := make([]uint32, n)
	for i := range out {
		v := s.pool[i%len(s.pool)]
		v ^= h
		h *= multB
		v *= h
		v ^= v >> xshift
		out[i] = v
	}
	return out
}

// GenerateState64 returns n uint64 words of state, numpy's
// generate_state(n, np.uint64): pairs of uint32 words, low word first.
func (s *SeedSequence) GenerateState64(n int) []uint64 {
	w := s.GenerateState(2 * n)
	out := make([]uint64, n)
	for i := range out {
		out[i] = uint64(w[2*i]) | uint64(w[2*i+1])<<32
	}
	return out
}

// Spawn returns n child SeedSequences, numpy's spawn(n): each has this
// entropy and pool size, and this spawn key extended by its index.
func (s *SeedSequence) Spawn(n int) []*SeedSequence {
	if n < 0 {
		panic("random: n_children must be non-negative")
	}
	out := make([]*SeedSequence, n)
	for i := range out {
		key := append(append([]uint64(nil), s.spawnKeyInts...), s.nChildrenSpawned+uint64(i))
		out[i] = newSeedSequence(s.entropy, key, s.poolSize, 0)
	}
	s.nChildrenSpawned += uint64(n)
	return out
}
