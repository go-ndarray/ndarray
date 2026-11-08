package bench

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-ndarray/ndarray"
)

// The data. Every input is lo + (hi-lo)*frac((i+off)*gold), built in Go and,
// as the same expression, in NumPy; TestDataParity compares the SHA-256 of both
// sides' bytes. The explicit float64 conversions forbid a fused multiply-add,
// so the bytes are the same on arm64.

const gold = 0.6180339887498949

// ds is one argument of a row: the NumPy expression that builds it, and the Go
// values (flat, row-major; re,im interleaved when cplx) it must equal.
type ds struct {
	py    string
	data  []float64
	shape []int
	cplx  bool
}

func vec(lo, hi float64, n int, off float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		f := math.Mod(float64((float64(i)+off)*gold), 1)
		x[i] = lo + float64((hi-lo)*f)
	}
	return x
}

func vpy(lo, hi float64, n int, off float64) string {
	return fmt.Sprintf("(%v + (%v - %v) * np.mod((np.arange(%d, dtype=np.float64) + %v) * 0.6180339887498949, 1.0))", lo, hi, lo, n, off)
}

func dvec(lo, hi float64, n int, off float64) ds {
	return ds{py: vpy(lo, hi, n, off), data: vec(lo, hi, n, off), shape: []int{n}}
}

func dmat(lo, hi float64, r, c int, off float64) ds {
	d := dvec(lo, hi, r*c, off)
	d.py += fmt.Sprintf(".reshape(%d, %d)", r, c)
	d.shape = []int{r, c}
	return d
}

// dreshape is d with another shape.
func dreshape(d ds, shape ...int) ds {
	d.py += fmt.Sprintf(".reshape(%s)", pyShape(shape))
	d.shape = shape
	return d
}

func pyShape(s []int) string {
	out := ""
	for i, v := range s {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprint(v)
	}
	return out
}

// dcplx is a complex128 array of the given shape.
func dcplx(shape ...int) ds {
	size := 1
	for _, v := range shape {
		size *= v
	}
	re, im := vec(-1, 1, size, 0), vec(-1, 1, size, 7)
	d := make([]float64, 2*size)
	for i := range re {
		d[2*i], d[2*i+1] = re[i], im[i]
	}
	sh := pyShape(shape)
	if len(shape) == 1 {
		sh += ","
	}
	return ds{
		py: fmt.Sprintf("(lambda r, i: np.stack([r, i], -1).view(np.complex128)[..., 0])(%s.reshape(%s), %s.reshape(%s))",
			vpy(-1, 1, size, 0), sh, vpy(-1, 1, size, 7), sh),
		data: d, shape: shape, cplx: true,
	}
}

// Square matrices of side n for linalg. Dominant: G + n·I (well conditioned,
// invertible). SPD: G + Gᵀ + 2n·I.
func dgen(n int) ds { return dmat(0, 1, n, n, 0) }

func ddom(n int) ds {
	g := dgen(n)
	d := slices.Clone(g.data)
	for i := 0; i < n; i++ {
		d[i*n+i] += float64(n)
	}
	return ds{py: fmt.Sprintf("(%s + %d * np.eye(%d))", g.py, n, n), data: d, shape: g.shape}
}

func dspd(n int) ds {
	g := dgen(n)
	d := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d[i*n+j] = g.data[i*n+j] + g.data[j*n+i]
			if i == j {
				d[i*n+j] += float64(2 * n)
			}
		}
	}
	return ds{py: fmt.Sprintf("(%s + %s.T + %d * np.eye(%d))", g.py, g.py, 2*n, n), data: d, shape: g.shape}
}

func (d ds) arr() *ndarray.Array {
	if d.cplx {
		c := make([]complex128, len(d.data)/2)
		for i := range c {
			c[i] = complex(d.data[2*i], d.data[2*i+1])
		}
		a, err := ndarray.FromSlice(c, d.shape...)
		if err != nil {
			panic(err)
		}
		return a
	}
	a, err := ndarray.FromData(slices.Clone(d.data), d.shape...)
	if err != nil {
		panic(err)
	}
	return a
}
