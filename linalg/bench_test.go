package linalg

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// benchMatrix is a seeded n×n standard normal matrix; spd makes it
// symmetric positive definite (A·Aᵀ + n·I), sym symmetric.
func benchMatrix(n int, kind string) *ndarray.Array {
	r := rand.New(rand.NewPCG(uint64(n), 7))
	d := make([]float64, n*n)
	for i := range d {
		d[i] = r.NormFloat64()
	}
	a, _ := ndarray.FromSlice(d, n, n)
	switch kind {
	case "spd":
		s, _ := a.MatMul(a.Transpose())
		for i := 0; i < n; i++ {
			s.Set(s.At(i, i)+float64(n), i, i)
		}
		return s
	case "sym":
		s, _ := a.Add(a.Transpose())
		return s
	}
	return a
}

var benchSizes = []int{64, 256, 1024}

// The operations measured against numpy in docs/linalg.md.
var benchOps = []struct {
	name, kind string
	f          func(a *ndarray.Array) error
}{
	{"solve", "", func(a *ndarray.Array) error {
		b, _ := ndarray.Ones(a.Shape()[0])
		_, err := Solve(a, b)
		return err
	}},
	{"inv", "", func(a *ndarray.Array) error { _, err := Inv(a); return err }},
	{"cholesky", "spd", func(a *ndarray.Array) error { _, err := Cholesky(a, Lower); return err }},
	{"qr", "", func(a *ndarray.Array) error { _, _, err := QR(a, ModeReduced); return err }},
	{"svd", "", func(a *ndarray.Array) error { _, _, _, err := SVD(a, false); return err }},
	{"eigh", "sym", func(a *ndarray.Array) error { _, _, err := Eigh(a, Lower); return err }},
	{"eig", "", func(a *ndarray.Array) error { _, _, err := Eig(a); return err }},
}

func BenchmarkLinalg(b *testing.B) {
	for _, op := range benchOps {
		for _, n := range benchSizes {
			a := benchMatrix(n, op.kind)
			b.Run(fmt.Sprintf("%s/%d", op.name, n), func(b *testing.B) {
				for b.Loop() {
					if err := op.f(a); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
