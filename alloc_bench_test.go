package ndarray

import "testing"

// The cost of a result, taken apart (#118): the allocation alone, the fill,
// and the operations built on them, at the size of the refbench table.

const allocN = 100_000

var (
	sinkF []float64
	sinkA *Array
)

func BenchmarkAllocMake(b *testing.B) {
	for b.Loop() {
		sinkF = make([]float64, allocN)
	}
}

func BenchmarkAllocZeros(b *testing.B) {
	for b.Loop() {
		sinkA, _ = Zeros(allocN)
	}
}

func BenchmarkAllocFull(b *testing.B) {
	for b.Loop() {
		sinkA, _ = Full(1.5, allocN)
	}
}

func BenchmarkFillLoop(b *testing.B) {
	d := make([]float64, allocN)
	for b.Loop() {
		for i := range d {
			d[i] = 1.5
		}
	}
	sinkF = d
}

func BenchmarkFillCopy(b *testing.B) {
	d := make([]float64, allocN)
	for b.Loop() {
		d[0] = 1.5
		for k := 1; k < len(d); k *= 2 {
			copy(d[k:], d[:k])
		}
	}
	sinkF = d
}

func BenchmarkAllocAdd(b *testing.B) {
	x, _ := Full(1.25, allocN)
	y, _ := Full(2.5, allocN)
	for b.Loop() {
		sinkA, _ = x.Add(y)
	}
}

func BenchmarkAllocAddScalar(b *testing.B) {
	x, _ := Full(1.25, allocN)
	for b.Loop() {
		sinkA = x.AddScalar(1.5)
	}
}
