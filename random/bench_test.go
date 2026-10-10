package random

import (
	"testing"

	"github.com/go-ndarray/ndarray"
)

// The benchmarks of docs/random.md: 10^6 draws each, as numpy's
// rng.random(10**6), rng.integers(0, 1000, 10**6),
// rng.standard_normal(10**6) and np.random.randn(10**6).

const benchN = 1_000_000

func BenchmarkRandom(b *testing.B) {
	g := DefaultRNG(0)
	for b.Loop() {
		g.Random(benchN)
	}
}

func BenchmarkRandomFloat32(b *testing.B) {
	g := DefaultRNG(0)
	for b.Loop() {
		g.RandomOf(ndarray.Float32, benchN)
	}
}

func BenchmarkIntegers(b *testing.B) {
	g := DefaultRNG(0)
	for b.Loop() {
		g.Integers(0, 1000, benchN)
	}
}

func BenchmarkStandardNormal(b *testing.B) {
	g := DefaultRNG(0)
	for b.Loop() {
		g.StandardNormal(benchN)
	}
}

func BenchmarkLegacyRandn(b *testing.B) {
	r := NewRandomState(0)
	for b.Loop() {
		r.Randn(benchN)
	}
}
