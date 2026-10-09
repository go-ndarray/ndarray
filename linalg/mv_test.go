package linalg

import (
	"testing"

	"github.com/go-ndarray/ndarray/internal/kernels"
)

func BenchmarkMatvec(b *testing.B) {
	n := 1024
	a := make([]float64, n*n)
	for i := range a {
		a[i] = float64(i%7) - 3
	}
	x, y := make([]float64, n), make([]float64, n)
	b.Run("ser", func(b *testing.B) {
		for b.Loop() {
			for r := 0; r < 1000; r++ {
				y[r] = kernels.Dot1DP(a[r*n:r*n+1000], x[:1000])
			}
		}
	})
	b.Run("pool", func(b *testing.B) {
		for b.Loop() {
			kernels.MatVecStridedP(y, a, n, x[:1000], 1000, 1000)
		}
	})
}
