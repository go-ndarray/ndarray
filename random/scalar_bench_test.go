package random

import (
	"strconv"
	"testing"
)

// Benchmarks of the scalar draws. NumPy's reference is the scalar call
// (rng.random(), rng.integers(0, n), rng.uniform(lo, hi, size=n),
// rng.shuffle(arange)), timed in a loop on the same machine; only run these
// when the one-minute load average is low.

var benchSink float64
var benchISink int64

func BenchmarkScalarFloat64(b *testing.B) {
	g := DefaultRNG(1)
	var s float64
	for range b.N {
		s += g.Float64()
	}
	benchSink = s
}

func BenchmarkScalarIntN(b *testing.B) {
	for _, tc := range []struct {
		name string
		high int64
	}{{"small", 97}, {"2^33", 1 << 33}} {
		b.Run("gen/"+tc.name, func(b *testing.B) {
			g := DefaultRNG(1)
			var s int64
			for range b.N {
				s += g.IntN(tc.high)
			}
			benchISink = s
		})
		b.Run("legacy/"+tc.name, func(b *testing.B) {
			r := NewRandomState(1)
			var s int64
			for range b.N {
				s += r.IntN(tc.high)
			}
			benchISink = s
		})
	}
}

func BenchmarkScalarFillUniform(b *testing.B) {
	dst := make([]float64, 30)
	g := DefaultRNG(1)
	r := NewRandomState(1)
	b.Run("gen/30", func(b *testing.B) {
		for range b.N {
			_ = g.FillUniform(dst, -5, 5)
		}
	})
	b.Run("legacy/30", func(b *testing.B) {
		for range b.N {
			_ = r.FillUniform(dst, -5, 5)
		}
	})
}

func BenchmarkScalarFillIntN(b *testing.B) {
	dst := make([]int64, 30)
	g := DefaultRNG(1)
	for range b.N {
		_ = g.FillIntN(dst, 30)
	}
	benchISink = dst[0]
}

func BenchmarkScalarShuffleInt64(b *testing.B) {
	for _, n := range []int{30, 300} {
		d := make([]int64, n)
		for i := range d {
			d[i] = int64(i)
		}
		b.Run("gen/"+strconv.Itoa(n), func(b *testing.B) {
			g := DefaultRNG(1)
			for range b.N {
				g.ShuffleInt64(d)
			}
		})
		b.Run("legacy/"+strconv.Itoa(n), func(b *testing.B) {
			r := NewRandomState(1)
			for range b.N {
				r.ShuffleInt64(d)
			}
		})
	}
}
