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

// BenchmarkStacked measures stacks of small matrices, which run one matrix
// per goroutine (numpy's gufuncs loop over them on one core).
func BenchmarkStacked(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 9))
	for _, sh := range [][2]int{{10000, 4}, {1000, 32}} {
		cnt, n := sh[0], sh[1]
		d := make([]float64, cnt*n*n)
		for i := range d {
			d[i] = r.NormFloat64()
		}
		a, _ := ndarray.FromSlice(d, cnt, n, n)
		s, _ := a.Add(swapLast(a))
		for _, op := range []struct {
			name string
			f    func() error
		}{
			{"inv", func() error { _, err := Inv(a); return err }},
			{"det", func() error { _, err := Det(a); return err }},
			{"eigh", func() error { _, _, err := Eigh(s, Lower); return err }},
			{"svd", func() error { _, _, _, err := SVD(a, false); return err }},
			{"eig", func() error { _, _, err := Eig(a); return err }},
		} {
			b.Run(fmt.Sprintf("%s/%dx%d", op.name, cnt, n), func(b *testing.B) {
				for b.Loop() {
					if err := op.f(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// swapLast transposes the last two axes of a 3-D array.
func swapLast(a *ndarray.Array) *ndarray.Array {
	s := a.Shape()
	d, _ := ndarray.Data[float64](a)
	out := make([]float64, len(d))
	m := s[1]
	for b := 0; b < s[0]; b++ {
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				out[b*m*m+j*m+i] = d[b*m*m+i*m+j]
			}
		}
	}
	t, _ := ndarray.FromSlice(out, s...)
	return t
}
