package benchmarks

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	nd "github.com/go-ndarray/ndarray"
)

// TestDumpForVerify writes go-ndarray results for a fixed set of ops to
// /tmp/gond_verify.txt so verify_numpy.py can confirm numerical agreement with
// NumPy on the identical inputs before any timing is trusted. It is a no-op
// unless GOND_VERIFY=1 (so it never runs in normal `go test`).
func TestDumpForVerify(t *testing.T) {
	if os.Getenv("GOND_VERIFY") != "1" {
		t.Skip("set GOND_VERIFY=1 to dump verification data")
	}
	f, err := os.Create("/tmp/gond_verify.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	put := func(tag string, v float64) { fmt.Fprintf(w, "%s\t%.17g\n", tag, v) }
	putArr := func(tag string, a *nd.Array) {
		c := a.Copy()
		fmt.Fprintf(w, "%s\tshape=%v\n", tag, c.Shape())
		flat := c.Ravel()
		// emit a few sampled elements + a checksum (sum) for cheap full check.
		put(tag+".sum", flat.Sum())
		put(tag+".first", flat.At(0))
		put(tag+".last", flat.At(flat.Size()-1))
		put(tag+".mid", flat.At(flat.Size()/2))
	}

	x, y := vec(1 << 16), vec(1 << 16)
	r, _ := x.Add(y)
	putArr("add", r)
	r, _ = x.Mul(y)
	putArr("mul", r)
	putArr("sqrt", x.Sqrt())
	putArr("exp", x.Exp())
	put("sum", x.Sum())
	mn, _ := x.Mean()
	put("mean", mn)
	mx, _ := x.Max()
	put("max", mx)

	M := mat(256, 256)
	N := mat(256, 256)
	mm, _ := M.MatMul(N)
	putArr("matmul", mm)

	R1, R2 := mat(300, 128), mat(128, 220)
	mm2, _ := R1.MatMul(R2)
	putArr("matmul_ns", mm2)

	s0, _ := M.SumAxis(0, false)
	putArr("sumaxis0", s0)
	s1, _ := M.SumAxis(1, false)
	putArr("sumaxis1", s1)
	mxa, _ := M.MaxAxis(1, false)
	putArr("maxaxis1", mxa)

	row := mat(1, 256)
	ba, _ := M.Add(row)
	putArr("bcast", ba)

	d, _ := x.Dot(y)
	put("dot", d.At())

	u, w2 := vec(700), vec(900)
	putArr("outer", u.Outer(w2))

	I1, I2 := mat(100, 64), mat(80, 64)
	in, _ := I1.Inner(I2)
	putArr("inner", in)

	v, _ := M.Slice(nd.Rng(0, 256, 2), nd.R(50, 200))
	putArr("slice", v)
}
