package random

import "math"

// Allocation-free scalar draws.
//
// Every sampler of Generator and RandomState returns a *ndarray.Array, even
// for one value; a solver that needs a handful of draws per candidate would
// spend more time allocating than evaluating. The methods here draw into
// caller storage or return a Go scalar, consume exactly the bit-generator
// output NumPy consumes for the same call, and allocate nothing on their
// success path.
//
// Sources (NumPy v2.5.3, opened): numpy/random/_generator.pyx (uniform,
// shuffle, _shuffle_raw), mtrand.pyx (uniform, randint, shuffle, _shuffle_raw),
// _bounded_integers.pyx.in (_rand_int64: scalar and sized paths both call
// random_bounded_uint64_fill) and src/distributions/distributions.c
// (random_uniform, random_interval, random_bounded_uint64_fill).

// uniformRange is the range numpy's uniform(low, high) passes to
// random_uniform. Both classes reject a range that is not finite
// (OverflowError); Generator also rejects a negative one (ValueError), the
// legacy RandomState accepts high < low.
func uniformRange(low, high float64, legacy bool) (float64, error) {
	rng := high - low
	if math.IsInf(rng, 0) || math.IsNaN(rng) {
		if legacy {
			return 0, valueErr("Range exceeds valid bounds")
		}
		return 0, valueErr("high - low range exceeds valid bounds")
	}
	if !legacy {
		if err := nonNegative(rng, "high - low"); err != nil {
			return 0, err
		}
	}
	return rng, nil
}

// fillUniform is random_uniform in a loop: low + rng*next_double. The
// float64 conversion forbids a fused multiply-add, which NumPy's x86-64
// builds do not use.
func fillUniform(bg BitGenerator, dst []float64, low, rng float64) {
	if p, ok := bg.(*PCG64); ok {
		for i := range dst {
			dst[i] = low + float64(rng*uint64ToDouble(p.Uint64()))
		}
		return
	}
	for i := range dst {
		dst[i] = low + float64(rng*bg.Float64())
	}
}

// intN is _rand_int64(0, high) for one value: random_bounded_uint64_fill
// with cnt = 1 on the range [0, high-1].
func intN(bg BitGenerator, high int64, masked bool) int64 {
	if high <= 0 {
		panic(boundsErr(false, 0))
	}
	rng := uint64(high - 1)
	if !masked && rng != 0 && rng < 0xFFFFFFFF {
		if p, ok := bg.(*PCG64); ok {
			return int64(lemire32PCG(p, uint32(rng)))
		}
	}
	return int64(boundedUint64(bg, 0, rng, masked))
}

// fillIntN is _rand_int64(0, high, size=len(dst)). An empty dst draws
// nothing and is never an error, as numpy returns an empty array first.
func fillIntN(bg BitGenerator, dst []int64, high int64, masked bool) error {
	if len(dst) == 0 {
		return nil
	}
	if high <= 0 {
		return boundsErr(false, 0)
	}
	fillUint64(bg, 0, uint64(high-1), masked, asUint64(dst))
	return nil
}

// shuffleInterval is _shuffle_raw with first = 1 (and the legacy
// _shuffle_raw, which is the same loop): Fisher-Yates from the last element
// down to the second, each j from random_interval(i).
func shuffleInterval(bg BitGenerator, d []int64) {
	if p, ok := bg.(*PCG64); ok {
		for i := len(d) - 1; i >= 1; i-- {
			j := intervalPCG(p, uint64(i))
			d[i], d[j] = d[j], d[i]
		}
		return
	}
	for i := len(d) - 1; i >= 1; i-- {
		j := interval(bg, uint64(i))
		d[i], d[j] = d[j], d[i]
	}
}

// intervalPCG is interval on a *PCG64, which the compiler can inline.
func intervalPCG(p *PCG64, max uint64) uint64 {
	mask := genMask(max)
	if max <= 0xFFFFFFFF {
		for {
			if v := uint64(p.Uint32()) & mask; v <= max {
				return v
			}
		}
	}
	for {
		if v := p.Uint64() & mask; v <= max {
			return v
		}
	}
}

// Float64 returns the next double in [0, 1), numpy's random() with
// size=None: next_double, 53 bits of one 64-bit output.
func (g *Generator) Float64() float64 { return g.bg.Float64() }

// FillUniform fills dst with values in [low, high), numpy's
// uniform(low, high, size=len(dst)) in order: low + (high-low)*U. It returns
// an error, and draws nothing, when high-low is NaN, infinite or negative.
func (g *Generator) FillUniform(dst []float64, low, high float64) error {
	rng, err := uniformRange(low, high, false)
	if err != nil {
		return err
	}
	fillUniform(g.bg, dst, low, rng)
	return nil
}

// IntN returns an integer in [0, high), numpy's integers(0, high) with the
// default int64 dtype and size=None: Lemire's bounded method, 32-bit draws
// when high-1 fits 32 bits (half of a 64-bit PCG64 output, the other half
// kept for the next 32-bit draw). IntN panics when high <= 0, as
// math/rand's Intn does; numpy raises ValueError.
func (g *Generator) IntN(high int64) int64 { return intN(g.bg, high, false) }

// FillIntN fills dst with integers in [0, high), numpy's
// integers(0, high, size=len(dst)). It returns an error, and draws nothing,
// when high <= 0 and dst is not empty.
func (g *Generator) FillIntN(dst []int64, high int64) error {
	return fillIntN(g.bg, dst, high, false)
}

// ShuffleInt64 shuffles d in place, numpy's shuffle of a one-dimensional
// int64 array: Fisher-Yates from the end, each index from random_interval.
func (g *Generator) ShuffleInt64(d []int64) { shuffleInterval(g.bg, d) }

// Float64 returns the next double in [0, 1), numpy's random_sample() with
// size=None.
func (r *RandomState) Float64() float64 { return r.bg.Float64() }

// FillUniform fills dst with values in [low, high), numpy's legacy
// uniform(low, high, size=len(dst)). Unlike Generator's it accepts
// high < low; it returns an error, and draws nothing, when high-low is NaN
// or infinite.
func (r *RandomState) FillUniform(dst []float64, low, high float64) error {
	rng, err := uniformRange(low, high, true)
	if err != nil {
		return err
	}
	fillUniform(r.bg, dst, low, rng)
	return nil
}

// IntN returns an integer in [0, high), numpy's legacy randint(0, high) with
// size=None, by masked rejection. It panics when high <= 0, as numpy raises
// ValueError.
func (r *RandomState) IntN(high int64) int64 { return intN(r.bg, high, true) }

// FillIntN fills dst with integers in [0, high), numpy's legacy
// randint(0, high, size=len(dst)). It returns an error, and draws nothing,
// when high <= 0 and dst is not empty.
func (r *RandomState) FillIntN(dst []int64, high int64) error {
	return fillIntN(r.bg, dst, high, true)
}

// ShuffleInt64 shuffles d in place, numpy's legacy shuffle of a
// one-dimensional int64 array.
func (r *RandomState) ShuffleInt64(d []int64) { shuffleInterval(r.bg, d) }
