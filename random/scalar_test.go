package random

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// scalar.json holds NumPy 2.5.3's output for the scalar draws, made by
// testdata/scalar.py on an x86-64 NumPy (TestRegenerateScalarFixture).
//
//go:embed testdata/scalar.json
var scalarJSON []byte

type scalarCall struct {
	Op string    `json:"op"`
	A  []string  `json:"a"`
	R  *[]string `json:"r"`
}

type scalarCase struct {
	API   string       `json:"api"`
	BG    string       `json:"bg"`
	Seed  string       `json:"seed"`
	Calls []scalarCall `json:"calls"`
}

type scalarFixture struct {
	NumPy   string       `json:"numpy"`
	Machine string       `json:"machine"`
	Cases   []scalarCase `json:"cases"`
}

func loadScalar(t testing.TB) scalarFixture {
	t.Helper()
	var f scalarFixture
	if err := json.Unmarshal(scalarJSON, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// drawer is what Generator and RandomState share.
type drawer interface {
	Float64() float64
	FillUniform(dst []float64, low, high float64) error
	IntN(high int64) int64
	FillIntN(dst []int64, high int64) error
	ShuffleInt64(d []int64)
}

var (
	_ drawer = (*Generator)(nil)
	_ drawer = (*RandomState)(nil)
)

func newDrawer(t testing.TB, api, bg, seed string) drawer {
	t.Helper()
	n, err := strconv.ParseUint(seed, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	var b BitGenerator
	switch {
	case api == "legacy" && bg == "MT19937":
		return NewRandomState(uint32(n))
	case bg == "PCG64":
		b = NewPCG64FromSeedSequence(NewSeedSequence(n))
	default:
		b = NewMT19937FromSeedSequence(NewSeedSequence(n))
	}
	if api == "gen" {
		return NewGenerator(b)
	}
	return NewRandomStateFrom(b)
}

func parseF(t testing.TB, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func parseI(t testing.TB, s string) int64 {
	t.Helper()
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func fstr(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

// intNOrPanic is IntN with its panic turned into ok = false.
func intNOrPanic(d drawer, high int64) (v int64, ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	return d.IntN(high), true
}

// runScalar replays one fixture call and returns what it produced, nil for
// an error (which NumPy raised too).
func runScalar(t testing.TB, d drawer, c scalarCall) []string {
	t.Helper()
	switch c.Op {
	case "f":
		return []string{strconv.FormatFloat(d.Float64(), 'g', -1, 64)}
	case "u":
		dst := make([]float64, parseI(t, c.A[2]))
		if d.FillUniform(dst, parseF(t, c.A[0]), parseF(t, c.A[1])) != nil {
			return nil
		}
		out := []string{}
		for _, v := range dst {
			out = append(out, fstr(v))
		}
		return out
	case "i":
		v, ok := intNOrPanic(d, parseI(t, c.A[0]))
		if !ok {
			return nil
		}
		return []string{strconv.FormatInt(v, 10)}
	case "fi":
		dst := make([]int64, parseI(t, c.A[1]))
		if d.FillIntN(dst, parseI(t, c.A[0])) != nil {
			return nil
		}
		out := []string{}
		for _, v := range dst {
			out = append(out, strconv.FormatInt(v, 10))
		}
		return out
	case "sh":
		x := make([]int64, parseI(t, c.A[0]))
		for i := range x {
			x[i] = int64(i)
		}
		d.ShuffleInt64(x)
		out := []string{}
		for _, v := range x {
			out = append(out, strconv.FormatInt(v, 10))
		}
		return out
	}
	t.Fatalf("unknown op %q", c.Op)
	return nil
}

// sameFloats compares float reprs by value (so "1e+300" and "1e300" agree).
func sameFloats(t testing.TB, got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		g, w := parseF(t, got[i]), parseF(t, want[i])
		if math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
			return false
		}
	}
	return true
}

func TestScalarDrawsMatchNumPy(t *testing.T) {
	f := loadScalar(t)
	if f.NumPy != "2.5.3" {
		t.Fatalf("fixture made by NumPy %s", f.NumPy)
	}
	if len(f.Cases) != 4*48 {
		t.Fatalf("%d cases", len(f.Cases))
	}
	var checked, errs int
	for _, c := range f.Cases {
		d := newDrawer(t, c.API, c.BG, c.Seed)
		for k, call := range c.Calls {
			got := runScalar(t, d, call)
			checked++
			if call.R == nil {
				errs++
				if got != nil {
					t.Fatalf("%s/%s seed %s call %d %s%v: NumPy raised, got %v", c.API, c.BG, c.Seed, k, call.Op, call.A, got)
				}
				continue
			}
			ok := got != nil
			if call.Op == "u" || call.Op == "f" {
				ok = ok && sameFloats(t, got, *call.R)
			} else {
				ok = ok && slices.Equal(got, *call.R)
			}
			if !ok {
				t.Fatalf("%s/%s seed %s call %d %s%v:\n got %v\nwant %v", c.API, c.BG, c.Seed, k, call.Op, call.A, got, *call.R)
			}
		}
	}
	t.Logf("%d calls on %d cases equal to NumPy %s, %d of them errors both sides", checked, len(f.Cases), f.NumPy, errs)
}

// TestScalarFixtureCoversEveryOp: the fixture must contain every kind of
// call, errors and successes, or the equality above proves less than it says.
func TestScalarFixtureCoversEveryOp(t *testing.T) {
	f := loadScalar(t)
	type key struct {
		api, bg, op string
		err         bool
	}
	seen := map[key]int{}
	for _, c := range f.Cases {
		for _, call := range c.Calls {
			seen[key{c.API, c.BG, call.Op, call.R == nil}]++
		}
	}
	for _, api := range [][2]string{{"gen", "PCG64"}, {"gen", "MT19937"}, {"legacy", "MT19937"}, {"legacy", "PCG64"}} {
		for _, op := range []string{"f", "u", "i", "fi", "sh"} {
			if seen[key{api[0], api[1], op, false}] < 20 {
				t.Errorf("%v %s: %d successes", api, op, seen[key{api[0], api[1], op, false}])
			}
		}
		for _, op := range []string{"u", "i", "fi"} {
			if seen[key{api[0], api[1], op, true}] < 5 {
				t.Errorf("%v %s: %d errors", api, op, seen[key{api[0], api[1], op, true}])
			}
		}
	}
}

// TestScalarDrawsDetectAWrongDraw plants breaks: a harness that cannot fail
// proves nothing. Each mutation of the stream must be caught by the replay.
func TestScalarDrawsDetectAWrongDraw(t *testing.T) {
	f := loadScalar(t)
	c := f.Cases[0] // Generator over PCG64, seed 0
	replay := func(skip int, wrap func(d drawer) drawer) bool {
		d := wrap(newDrawer(t, c.API, c.BG, c.Seed))
		for k, call := range c.Calls {
			if k == skip {
				continue
			}
			got := runScalar(t, d, call)
			if call.R == nil {
				if got != nil {
					return false
				}
				continue
			}
			if got == nil || len(got) != len(*call.R) {
				return false
			}
			if call.Op == "u" || call.Op == "f" {
				if !sameFloats(t, got, *call.R) {
					return false
				}
			} else if !slices.Equal(got, *call.R) {
				return false
			}
		}
		return true
	}
	if !replay(-1, func(d drawer) drawer { return d }) {
		t.Fatal("control: the unmodified replay must pass")
	}
	if replay(0, func(d drawer) drawer { return d }) {
		t.Error("a skipped call (a draw swap) went unnoticed")
	}
	if replay(-1, func(d drawer) drawer { d.Float64(); return d }) {
		t.Error("an extra leading draw went unnoticed")
	}
	g := NewGenerator(NewPCG64FromSeedSequence(NewSeedSequence(0)))
	if replay(-1, func(drawer) drawer { return &RandomState{bg: g.bg} }) {
		t.Error("masked instead of Lemire went unnoticed")
	}
}

// TestRegenerateScalarFixture rebuilds testdata/scalar.json with an x86-64
// NumPy 2.5.3 in a container (this machine has no Rosetta; the container runs
// the amd64 image under emulation). GONDARRAY_REGEN=1 enables it.
func TestRegenerateScalarFixture(t *testing.T) {
	if os.Getenv("GONDARRAY_REGEN") != "1" {
		t.Skip("set GONDARRAY_REGEN=1 to rebuild testdata/scalar.json")
	}
	dir, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd := exec.Command("docker", "run", "--rm", "--platform", "linux/amd64",
		"-v", dir+":/w", "-w", "/w", "--entrypoint", "uv",
		"ghcr.io/astral-sh/uv:0.9.5-python3.12-bookworm-slim",
		"run", "--no-project", "--with", "numpy==2.5.3", "python", "scalar.py", "/w/scalar.json")
	cmd.Stderr = &stderr
	cmd.Stdin = nil
	if err := cmd.Run(); err != nil {
		t.Fatalf("%v\n%s", err, stderr.String())
	}
}

func TestScalarErrors(t *testing.T) {
	g := DefaultRNG(1)
	r := NewRandomState(1)
	nan, inf := math.NaN(), math.Inf(1)
	dst := make([]float64, 3)
	for _, c := range [][2]float64{{0, nan}, {nan, 0}, {0, inf}, {-inf, 0}, {inf, inf}, {1, 0}, {1e308, -1e308}} {
		if g.FillUniform(dst, c[0], c[1]) == nil {
			t.Errorf("Generator.FillUniform(%v, %v) must fail", c[0], c[1])
		}
	}
	for _, c := range [][2]float64{{0, nan}, {nan, 0}, {0, inf}, {-inf, 0}, {inf, inf}, {1e308, -1e308}, {-1.7e308, 1.7e308}} {
		if r.FillUniform(dst, c[0], c[1]) == nil {
			t.Errorf("RandomState.FillUniform(%v, %v) must fail", c[0], c[1])
		}
	}
	if err := r.FillUniform(dst, 1, 0); err != nil {
		t.Errorf("legacy uniform accepts high < low: %v", err)
	}
	ints := make([]int64, 2)
	for _, h := range []int64{0, -1, math.MinInt64} {
		if err := g.FillIntN(ints, h); !errors.Is(err, ErrValue) {
			t.Errorf("Generator.FillIntN(%d) = %v", h, err)
		}
		if err := r.FillIntN(ints, h); !errors.Is(err, ErrValue) {
			t.Errorf("RandomState.FillIntN(%d) = %v", h, err)
		}
		if err := g.FillIntN(nil, h); err != nil {
			t.Errorf("an empty dst is never an error: %v", err)
		}
		if _, ok := intNOrPanic(g, h); ok {
			t.Errorf("IntN(%d) must panic", h)
		}
		if _, ok := intNOrPanic(r, h); ok {
			t.Errorf("legacy IntN(%d) must panic", h)
		}
	}
	// Nothing was consumed by the failures.
	if a, b := g.Float64(), DefaultRNG(1).Float64(); a != b {
		t.Errorf("failed calls consumed draws: %v != %v", a, b)
	}
	g.ShuffleInt64(nil)
	g.ShuffleInt64([]int64{4})
}

// TestScalarMatchesArrayAPI: the scalar draws consume the stream exactly as
// the array samplers do, on a bit generator other than the ones the fixture
// reaches (Philox, SFC64, PCG64DXSM), where the only reference is this
// package's own array code, itself checked against NumPy.
func TestScalarMatchesArrayAPI(t *testing.T) {
	mk := map[string]func() BitGenerator{
		"Philox":    func() BitGenerator { return NewPhiloxFromSeedSequence(NewSeedSequence(3)) },
		"SFC64":     func() BitGenerator { return NewSFC64FromSeedSequence(NewSeedSequence(3)) },
		"PCG64DXSM": func() BitGenerator { return NewPCG64DXSMFromSeedSequence(NewSeedSequence(3)) },
		"PCG64":     func() BitGenerator { return NewPCG64FromSeedSequence(NewSeedSequence(3)) },
	}
	for name, f := range mk {
		a, b := NewGenerator(f()), NewGenerator(f())
		for _, high := range []int64{1, 7, 1 << 33, math.MaxInt64} {
			arr, err := a.Integers(0, high, 5)
			if err != nil {
				t.Fatal(err)
			}
			want, _ := ndarray.Data[int64](arr)
			got := make([]int64, 5)
			if err := b.FillIntN(got, high); err != nil || !slices.Equal(got, want) {
				t.Errorf("%s FillIntN(%d) = %v, %v; want %v", name, high, got, err, want)
			}
			one, err := a.Integers(0, high)
			if err != nil {
				t.Fatal(err)
			}
			w1, _ := ndarray.Data[int64](one)
			if v := b.IntN(high); v != w1[0] {
				t.Errorf("%s IntN(%d) = %d, want %d", name, high, v, w1[0])
			}
		}
		u, err := a.Uniform(-2, 3, 4)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := ndarray.Data[float64](u)
		du := make([]float64, 4)
		if err := b.FillUniform(du, -2, 3); err != nil || !slices.Equal(du, want) {
			t.Errorf("%s FillUniform = %v, %v; want %v", name, du, err, want)
		}
		x, err := a.Permutation(40)
		if err != nil {
			t.Fatal(err)
		}
		wp, _ := ndarray.Data[int64](x)
		y := make([]int64, 40)
		for i := range y {
			y[i] = int64(i)
		}
		b.ShuffleInt64(y)
		if !slices.Equal(y, wp) {
			t.Errorf("%s ShuffleInt64 differs from Permutation", name)
		}
	}
}

func TestScalarAllocations(t *testing.T) {
	gens := map[string]drawer{
		"gen/PCG64":    DefaultRNG(5),
		"gen/MT19937":  NewGenerator(NewMT19937FromSeedSequence(NewSeedSequence(5))),
		"legacy/MT":    NewRandomState(5),
		"legacy/PCG64": NewRandomStateFrom(NewPCG64FromSeedSequence(NewSeedSequence(5))),
	}
	fs := make([]float64, 16)
	is := make([]int64, 16)
	sh := make([]int64, 64)
	var sink float64
	var isink int64
	for name, d := range gens {
		for what, f := range map[string]func(){
			"Float64":      func() { sink += d.Float64() },
			"FillUniform":  func() { _ = d.FillUniform(fs, -1, 2) },
			"IntN small":   func() { isink += d.IntN(97) },
			"IntN 2^33":    func() { isink += d.IntN(1 << 33) },
			"IntN 2^32":    func() { isink += d.IntN(1 << 32) },
			"IntN max":     func() { isink += d.IntN(math.MaxInt64) },
			"FillIntN":     func() { _ = d.FillIntN(is, 1000) },
			"FillIntN big": func() { _ = d.FillIntN(is, 1<<40) },
			"ShuffleInt64": func() { d.ShuffleInt64(sh) },
		} {
			if n := testing.AllocsPerRun(100, f); n != 0 {
				t.Errorf("%s %s: %v allocations per run", name, what, n)
			}
		}
	}
	_, _ = sink, isink
}

func TestShuffleIsAPermutation(t *testing.T) {
	for _, d := range []drawer{DefaultRNG(9), NewRandomState(9), NewGenerator(NewMT19937FromSeedSequence(NewSeedSequence(9)))} {
		x := make([]int64, 300)
		for i := range x {
			x[i] = int64(i)
		}
		d.ShuffleInt64(x)
		y := slices.Clone(x)
		slices.Sort(y)
		for i := range y {
			if y[i] != int64(i) {
				t.Fatalf("not a permutation at %d", i)
			}
		}
	}
}

// intervalWide covers random_interval's 64-bit branch (max beyond 32 bits),
// which a slice length cannot reach on a 32-bit target and a real shuffle
// never reaches at all.
func TestIntervalWideMatchesGeneric(t *testing.T) {
	for _, max := range []uint64{0xFFFFFFFF, 1 << 32, 1<<40 + 5, math.MaxUint64} {
		p := NewPCG64FromSeedSequence(NewSeedSequence(11))
		q := NewPCG64FromSeedSequence(NewSeedSequence(11))
		for range 50 {
			if a, b := intervalPCG(p, max), interval(q, max); a != b {
				t.Fatalf("max %d: %d != %d", max, a, b)
			}
		}
	}
}

func FuzzScalarDraws(f *testing.F) {
	nan, inf := math.NaN(), math.Inf(1)
	for _, s := range []struct {
		lo, hi float64
		h      int64
		n      uint8
	}{
		{0, 1, 10, 3}, {nan, 1, 0, 1}, {0, inf, -1, 0}, {-inf, inf, math.MaxInt64, 5},
		{1, 0, math.MinInt64, 2}, {1e308, -1e308, 1 << 32, 4}, {0, 5e-324, 1<<32 - 1, 7},
		{-0.0, 0.0, 1<<32 + 1, 9}, {1.7e308, 1.7e308, 1<<63 - 1, 64},
	} {
		f.Add(s.lo, s.hi, s.h, s.n)
	}
	f.Fuzz(func(t *testing.T, lo, hi float64, high int64, n uint8) {
		for _, d := range []drawer{DefaultRNG(1), NewRandomState(1)} {
			dst := make([]float64, n)
			err := d.FillUniform(dst, lo, hi)
			rng := hi - lo
			_, legacy := d.(*RandomState)
			wantErr := math.IsNaN(rng) || math.IsInf(rng, 0) || (!legacy && math.Signbit(rng) && !math.IsNaN(rng))
			if (err != nil) != wantErr {
				t.Fatalf("FillUniform(%v, %v): err = %v", lo, hi, err)
			}
			if err == nil {
				for _, v := range dst {
					if !legacy && (v < lo || v > hi) && !math.IsNaN(v) {
						// high may be reached by rounding, never exceeded
						if v > hi || v < lo {
							t.Fatalf("%v outside [%v, %v]", v, lo, hi)
						}
					}
				}
			}
			ints := make([]int64, n)
			err = d.FillIntN(ints, high)
			if (err != nil) != (high <= 0 && n > 0) {
				t.Fatalf("FillIntN(%d) n=%d: err = %v", high, n, err)
			}
			if err == nil && n > 0 {
				for _, v := range ints {
					if v < 0 || v >= high {
						t.Fatalf("FillIntN(%d) = %d", high, v)
					}
				}
			}
			v, ok := intNOrPanic(d, high)
			if ok != (high > 0) || (ok && (v < 0 || v >= high)) {
				t.Fatalf("IntN(%d) = %d, %v", high, v, ok)
			}
			d.ShuffleInt64(ints)
		}
	})
}
