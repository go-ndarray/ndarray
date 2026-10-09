package random

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// cases.json holds NumPy's output for every function on several seeds and
// bit generators, made by testdata/cases.py with an x86-64 NumPy.
//
//go:embed testdata/cases.json
var casesJSON []byte

type encoded struct {
	DType string   `json:"dtype"`
	Shape []int    `json:"shape"`
	V     []string `json:"v"`
}

type call struct {
	Fn   string         `json:"fn"`
	Kw   map[string]any `json:"kw"`
	Size any            `json:"size"`
	R    encoded        `json:"r"`
}

type script struct {
	API   string `json:"api"`
	BG    string `json:"bg"`
	Seed  string `json:"seed"`
	Calls []call `json:"calls"`
}

type bitgenRec struct {
	BG         string     `json:"bg"`
	Seed       string     `json:"seed"`
	Raw        []string   `json:"raw"`
	State      []string   `json:"state"`
	Jumped1    []string   `json:"jumped1"`
	Jumped3    []string   `json:"jumped3"`
	Advance    []string   `json:"advance"`
	AdvanceNeg []string   `json:"advance_neg"`
	Spawn      [][]string `json:"spawn"`
}

type fixture struct {
	NumPy   string   `json:"numpy"`
	Machine string   `json:"machine"`
	Cases   []script `json:"cases"`
	Digests []struct {
		API  string         `json:"api"`
		Seed string         `json:"seed"`
		Fn   string         `json:"fn"`
		Kw   map[string]any `json:"kw"`
		Size any            `json:"size"`
		D    struct {
			DType  string `json:"dtype"`
			Shape  []int  `json:"shape"`
			SHA256 string `json:"sha256"`
		} `json:"d"`
	} `json:"digests"`
	Bitgens   []bitgenRec `json:"bitgens"`
	PhiloxKey struct {
		Raw   []string `json:"raw"`
		Carry []string `json:"carry"`
	} `json:"philox_key"`
	LegacyMT []struct {
		Seed        string   `json:"seed"`
		Raw         []string `json:"raw"`
		Jumped      []string `json:"jumped"`
		JumpedFresh []string `json:"jumped_fresh"`
	} `json:"legacy_mt"`
	SeedSeqs []struct {
		Name    string   `json:"name"`
		Pool    []string `json:"pool"`
		State32 []string `json:"state32"`
		State64 []string `json:"state64"`
		Kids    []struct {
			Key  []string `json:"key"`
			Pool []string `json:"pool"`
		} `json:"kids"`
	} `json:"seedseqs"`
}

var fx = func() *fixture {
	var f fixture
	dec := json.NewDecoder(bytes.NewReader(casesJSON))
	dec.UseNumber()
	if err := dec.Decode(&f); err != nil {
		panic(err)
	}
	return &f
}()

func bigOf(s string) *big.Int {
	v, ok := new(big.Int).SetString(s, 10)
	if !ok {
		panic(s)
	}
	return v
}

func u64(s string) uint64 { return bigOf(s).Uint64() }

// seedSeqs are the fixture's seeds, as NumPy's SeedSequence(seed).
var seedSeqs = map[string]func() *SeedSequence{
	"0":      func() *SeedSequence { return NewSeedSequence(0) },
	"1":      func() *SeedSequence { return NewSeedSequence(1) },
	"42":     func() *SeedSequence { return NewSeedSequence(42) },
	"u64max": func() *SeedSequence { return NewSeedSequence(math.MaxUint64) },
	"list":   func() *SeedSequence { return NewSeedSequence(1, 2, 3) },
	"big": func() *SeedSequence {
		ss, err := NewSeedSequenceBig(bigOf("340282366920938463463374607431768223801"))
		if err != nil {
			panic(err)
		}
		return ss
	},
}

func newBG(kind, seed string) BitGenerator {
	ss := seedSeqs[seed]()
	switch kind {
	case "PCG64":
		return NewPCG64FromSeedSequence(ss)
	case "PCG64DXSM":
		return NewPCG64DXSMFromSeedSequence(ss)
	case "MT19937":
		return NewMT19937FromSeedSequence(ss)
	case "Philox":
		return NewPhiloxFromSeedSequence(ss)
	}
	return NewSFC64FromSeedSequence(ss)
}

var legacySeeds = map[string]func() *RandomState{
	"0":      func() *RandomState { return NewRandomState(0) },
	"1":      func() *RandomState { return NewRandomState(1) },
	"42":     func() *RandomState { return NewRandomState(42) },
	"u32max": func() *RandomState { return NewRandomState(math.MaxUint32) },
	"arr":    func() *RandomState { r, _ := NewRandomStateArray([]uint32{1, 2, 3}); return r },
	"arr700": func() *RandomState {
		k := make([]uint32, 700)
		for i := range k {
			k[i] = uint32(i)
		}
		r, _ := NewRandomStateArray(k)
		return r
	},
}

func wantStrings(t *testing.T, what string, got []uint64, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: %d values, NumPy %d", what, len(got), len(want))
		return
	}
	for i := range want {
		if got[i] != u64(want[i]) {
			t.Errorf("%s[%d] = %d, NumPy %s", what, i, got[i], want[i])
			return
		}
	}
}

func TestSeedSequenceMatchesNumPy(t *testing.T) {
	mk := map[string]func() *SeedSequence{
		"0": seedSeqs["0"], "42": seedSeqs["42"], "u64max": seedSeqs["u64max"], "list": seedSeqs["list"],
		"big":   seedSeqs["big"],
		"long":  func() *SeedSequence { return NewSeedSequence(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) },
		"empty": func() *SeedSequence { s, _ := NewSeedSequenceWith(SeedSequenceOptions{Words: []uint32{}}); return s },
		"pool8": func() *SeedSequence {
			s, _ := NewSeedSequenceWith(SeedSequenceOptions{Entropy: []uint64{7}, PoolSize: 8})
			return s
		},
		"spawnkey": func() *SeedSequence {
			s, _ := NewSeedSequenceWith(SeedSequenceOptions{Entropy: []uint64{7}, SpawnKey: []uint64{3, 1 << 40}})
			return s
		},
		"spawnkey_long": func() *SeedSequence {
			s, _ := NewSeedSequenceWith(SeedSequenceOptions{Entropy: []uint64{0, 1, 2, 3, 4, 5}, SpawnKey: []uint64{1}})
			return s
		},
	}
	for _, c := range fx.SeedSeqs {
		ss := mk[c.Name]()
		wantStrings(t, c.Name+" pool", widen(ss.Pool()), c.Pool)
		wantStrings(t, c.Name+" state32", widen(ss.GenerateState(9)), c.State32)
		wantStrings(t, c.Name+" state64", ss.GenerateState64(5), c.State64)
		kids := append(ss.Spawn(3), ss.Spawn(1)...)
		for i, k := range kids {
			wantStrings(t, fmt.Sprintf("%s kid %d key", c.Name, i), k.SpawnKey(), c.Kids[i].Key)
			wantStrings(t, fmt.Sprintf("%s kid %d pool", c.Name, i), widen(k.Pool()), c.Kids[i].Pool)
		}
	}
}

func widen(w []uint32) []uint64 {
	out := make([]uint64, len(w))
	for i, v := range w {
		out[i] = uint64(v)
	}
	return out
}

func TestBitGeneratorsMatchNumPy(t *testing.T) {
	for _, r := range fx.Bitgens {
		name := r.BG + "/" + r.Seed
		bg := newBG(r.BG, r.Seed)
		var state []uint64
		switch b := bg.(type) {
		case *PCG64:
			s := b.State()
			state = []uint64{s.StateHi, s.StateLo, s.IncHi, s.IncLo}
		case *PCG64DXSM:
			s := b.State()
			state = []uint64{s.StateHi, s.StateLo, s.IncHi, s.IncLo}
		case *MT19937:
			s := b.State()
			state = []uint64{uint64(s.Key[0]), uint64(s.Key[1]), uint64(s.Key[2]), uint64(s.Key[623]), uint64(s.Pos)}
		case *Philox:
			s := b.State()
			state = s.Key[:]
		case *SFC64:
			s := b.State()
			state = s.S[:]
		}
		if r.BG == "PCG64" || r.BG == "PCG64DXSM" {
			st, inc := bigOf(r.State[0]), bigOf(r.State[1])
			want := []uint64{new(big.Int).Rsh(st, 64).Uint64(), st.Uint64(), new(big.Int).Rsh(inc, 64).Uint64(), inc.Uint64()}
			if fmt.Sprint(state) != fmt.Sprint(want) {
				t.Errorf("%s state %v, NumPy %v", name, state, want)
			}
		} else {
			wantStrings(t, name+" state", state, r.State)
		}
		wantStrings(t, name+" raw", RandomRaw(bg, len(r.Raw)), r.Raw)
		for j, want := range map[int][]string{1: r.Jumped1, 3: r.Jumped3} {
			if want == nil {
				continue
			}
			var jb BitGenerator
			switch b := newBG(r.BG, r.Seed).(type) {
			case *PCG64:
				jb = b.Jumped(uint64(j))
			case *PCG64DXSM:
				jb = b.Jumped(uint64(j))
			case *MT19937:
				jb = b.Jumped(j)
			case *Philox:
				jb = b.Jumped(uint64(j))
			}
			wantStrings(t, fmt.Sprintf("%s jumped(%d)", name, j), RandomRaw(jb, 5), want)
		}
		if r.Advance != nil {
			adv := func(skip int, d *big.Int) BitGenerator {
				b := newBG(r.BG, r.Seed)
				RandomRaw(b, skip)
				switch b := b.(type) {
				case *PCG64:
					return b.Advance(d)
				case *PCG64DXSM:
					return b.Advance(d)
				}
				return b.(*Philox).Advance(d)
			}
			d := new(big.Int).Lsh(big.NewInt(1), 100)
			wantStrings(t, name+" advance", RandomRaw(adv(3, d.Add(d, big.NewInt(7))), 5), r.Advance)
			wantStrings(t, name+" advance(-3)", RandomRaw(adv(7, big.NewInt(-3)), 5), r.AdvanceNeg)
		}
		kids, err := newBG(r.BG, r.Seed).(Spawner).Spawn(2)
		if err != nil {
			t.Fatal(err)
		}
		for i, k := range kids {
			wantStrings(t, fmt.Sprintf("%s spawn %d", name, i), RandomRaw(k, 5), r.Spawn[i])
		}
	}
	ph := NewPhiloxKey([4]uint64{1, 2, 3, math.MaxUint64}, [2]uint64{5, 6})
	wantStrings(t, "Philox(counter, key)", RandomRaw(ph, 12), fx.PhiloxKey.Raw)
	ph = NewPhiloxKey([4]uint64{math.MaxUint64, math.MaxUint64, 0, 0}, [2]uint64{1, 2}).Advance(big.NewInt(1))
	wantStrings(t, "Philox advance carry", RandomRaw(ph, 4), fx.PhiloxKey.Carry)
	for _, r := range fx.LegacyMT {
		m := legacySeeds[r.Seed]().BitGenerator().(*MT19937)
		wantStrings(t, "legacy MT19937 "+r.Seed, RandomRaw(m, len(r.Raw)), r.Raw)
		wantStrings(t, "legacy MT19937 jumped "+r.Seed, RandomRaw(m.Jumped(1), 5), r.Jumped)
		fresh := legacySeeds[r.Seed]().BitGenerator().(*MT19937)
		wantStrings(t, "legacy MT19937 jumped fresh "+r.Seed, RandomRaw(fresh.Jumped(1), 5), r.JumpedFresh)
	}
}

// libmULP bounds, per sampler, the distance in ULPs from NumPy's value of a
// result that passes through a C library function. This package's log,
// log1p, exp, expm1 and pow are correctly rounded (libm.go); NumPy's come
// from the C library, which is too in all but a fraction of a percent of
// arguments. The fixture holds a few such values (each checked against
// mpmath: NumPy's is the misrounded one); beta magnifies its one to 2 ULP.
// vonmises calls cos and acos, which neither side rounds correctly (Apple's
// cos misrounds 17% of arguments), and acos near ±1 magnifies an ULP.
//
// Every other sampler, every integer-valued one, and every value not listed
// here must match to the bit: a libm difference that flipped an
// accept/reject decision would change the stream and fail.
var libmULP = map[string]int64{
	"standard_exponential": 1, // the "inv" method is -log1p(-U)
	"standard_gamma":       2, // pow (shape < 1); log in the rejection test
	"beta":                 2,
	"lognormal":            1,
	"randn":                1,  // RandomState's polar method takes a log per pair
	"normal":               1,  // RandomState's, likewise
	"vonmises":             64, // in ULPs of pi: see absULP
}

// absULP lists the samplers whose differences are measured in ULPs of a
// fixed scale rather than of the value: vonmises wraps its angle into
// [-pi, pi], so a value near 0 carries the absolute error of one near pi.
var absULP = map[string]float64{"vonmises": math.Pi}

func sizeOf(v any) []int {
	switch s := v.(type) {
	case nil:
		return nil
	case json.Number:
		return []int{int(numF(s))}
	}
	var out []int
	for _, d := range v.([]any) {
		out = append(out, int(numF(d)))
	}
	return out
}

func kwF(kw map[string]any, k string, def float64) float64 {
	if v, ok := kw[k]; ok {
		return numF(v)
	}
	return def
}

func kwI(kw map[string]any, k string) int64 {
	return bigOf(string(kw[k].(json.Number))).Int64()
}

func kwDType(kw map[string]any, def ndarray.DType) ndarray.DType {
	if v, ok := kw["dtype"]; ok {
		dt, err := ndarray.ParseDType(v.(string))
		if err != nil {
			panic(err)
		}
		return dt
	}
	return def
}

func kwFloats(kw map[string]any, k string) []float64 {
	v, ok := kw[k].([]any)
	if !ok {
		return nil
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = numF(x)
	}
	return out
}

func namedArray(name string) *ndarray.Array {
	switch name {
	case "arange10":
		a, _ := ndarray.FromSlice([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 10)
		return a
	case "arange12":
		d := make([]int64, 12)
		for i := range d {
			d[i] = int64(i)
		}
		a, _ := ndarray.FromSlice(d, 3, 4)
		return a
	}
	a, _ := ndarray.FromSlice([]float64{0.5, 1.5, 2.5, 3.5, 4.5, 5.5}, 6)
	return a
}

func choiceOpts(kw map[string]any) ChoiceOptions {
	o := ChoiceOptions{P: kwFloats(kw, "p")}
	if r, ok := kw["replace"]; ok && !r.(bool) {
		o.NoReplace = true
	}
	if s, ok := kw["shuffle"]; ok && !s.(bool) {
		o.NoShuffle = true
	}
	if a, ok := kw["axis"]; ok {
		o.Axis = int(numF(a))
	}
	return o
}

func bytesArray(b []byte) *ndarray.Array {
	a, _ := ndarray.FromSlice(b, len(b))
	return a
}

// bigBounds handles integers bounds beyond int64 (uint64 dtype).
func bigBounds(kw map[string]any) (lo, hi *big.Int) {
	return bigOf(numString(kw["low"])), bigOf(numString(kw["high"]))
}

func numString(v any) string { return string(v.(json.Number)) }

func numF(v any) float64 {
	f, err := v.(json.Number).Float64()
	if err != nil {
		panic(err)
	}
	return f
}

func runGenerator(g *Generator, c call) (*ndarray.Array, error) {
	kw, size := c.Kw, sizeOf(c.Size)
	switch c.Fn {
	case "random":
		return g.RandomOf(kwDType(kw, ndarray.Float64), size...)
	case "integers":
		dt := kwDType(kw, ndarray.Int64)
		ep, _ := kw["endpoint"].(bool)
		if dt == ndarray.Uint64 {
			lo, hi := bigBounds(kw)
			return g.IntegersUint64(lo.Uint64(), hi.Uint64(), ep, size...)
		}
		lo, hi := bigBounds(kw)
		return g.IntegersOf(dt, lo.Int64(), hi.Int64(), ep, size...)
	case "standard_normal":
		return g.StandardNormalOf(kwDType(kw, ndarray.Float64), size...)
	case "standard_exponential":
		m, ok := kw["method"].(string)
		if !ok {
			m = "zig"
		}
		return g.StandardExponentialOf(kwDType(kw, ndarray.Float64), m, size...)
	case "standard_gamma":
		return g.StandardGammaOf(kwDType(kw, ndarray.Float64), kwF(kw, "shape", 0), size...)
	case "normal":
		return g.Normal(kwF(kw, "loc", 0), kwF(kw, "scale", 1), size...)
	case "exponential":
		return g.Exponential(kwF(kw, "scale", 1), size...)
	case "uniform":
		return g.Uniform(kwF(kw, "low", 0), kwF(kw, "high", 1), size...)
	case "gamma":
		return g.Gamma(kwF(kw, "shape", 0), kwF(kw, "scale", 1), size...)
	case "beta":
		return g.Beta(kwF(kw, "a", 0), kwF(kw, "b", 0), size...)
	case "chisquare":
		return g.ChiSquare(kwF(kw, "df", 0), size...)
	case "f":
		return g.F(kwF(kw, "dfnum", 0), kwF(kw, "dfden", 0), size...)
	case "noncentral_chisquare":
		return g.NoncentralChiSquare(kwF(kw, "df", 0), kwF(kw, "nonc", 0), size...)
	case "noncentral_f":
		return g.NoncentralF(kwF(kw, "dfnum", 0), kwF(kw, "dfden", 0), kwF(kw, "nonc", 0), size...)
	case "standard_cauchy":
		return g.StandardCauchy(size...)
	case "standard_t":
		return g.StandardT(kwF(kw, "df", 0), size...)
	case "vonmises":
		return g.VonMises(kwF(kw, "mu", 0), kwF(kw, "kappa", 0), size...)
	case "pareto":
		return g.Pareto(kwF(kw, "a", 0), size...)
	case "weibull":
		return g.Weibull(kwF(kw, "a", 0), size...)
	case "power":
		return g.Power(kwF(kw, "a", 0), size...)
	case "laplace":
		return g.Laplace(kwF(kw, "loc", 0), kwF(kw, "scale", 1), size...)
	case "gumbel":
		return g.Gumbel(kwF(kw, "loc", 0), kwF(kw, "scale", 1), size...)
	case "logistic":
		return g.Logistic(kwF(kw, "loc", 0), kwF(kw, "scale", 1), size...)
	case "lognormal":
		return g.Lognormal(kwF(kw, "mean", 0), kwF(kw, "sigma", 1), size...)
	case "rayleigh":
		return g.Rayleigh(kwF(kw, "scale", 1), size...)
	case "wald":
		return g.Wald(kwF(kw, "mean", 0), kwF(kw, "scale", 0), size...)
	case "triangular":
		return g.Triangular(kwF(kw, "left", 0), kwF(kw, "mode", 0), kwF(kw, "right", 0), size...)
	case "binomial":
		return g.Binomial(kwI(kw, "n"), kwF(kw, "p", 0), size...)
	case "negative_binomial":
		return g.NegativeBinomial(kwF(kw, "n", 0), kwF(kw, "p", 0), size...)
	case "poisson":
		return g.Poisson(kwF(kw, "lam", 1), size...)
	case "zipf":
		return g.Zipf(kwF(kw, "a", 0), size...)
	case "geometric":
		return g.Geometric(kwF(kw, "p", 0), size...)
	case "logseries":
		return g.Logseries(kwF(kw, "p", 0), size...)
	case "choice":
		if name, ok := kw["a"].(string); ok {
			return g.Choice(namedArray(name), choiceOpts(kw), size...)
		}
		return g.ChoiceN(kwI(kw, "a"), choiceOpts(kw), size...)
	case "shuffle":
		a := namedArray(kw["x"].(string))
		axis := 0
		if v, ok := kw["axis"]; ok {
			axis = int(numF(v))
		}
		return a, g.Shuffle(a, axis)
	case "permutation":
		if name, ok := kw["x"].(string); ok {
			axis := 0
			if v, ok := kw["axis"]; ok {
				axis = int(numF(v))
			}
			return g.PermutationOf(namedArray(name), axis)
		}
		return g.Permutation(int(kwI(kw, "x")))
	case "permuted":
		a := namedArray(kw["x"].(string))
		if v, ok := kw["axis"]; ok {
			return g.PermutedAxis(a, int(numF(v)))
		}
		return g.Permuted(a)
	case "bytes":
		return bytesArray(g.Bytes(int(kwI(kw, "length")))), nil
	case "dirichlet":
		return g.Dirichlet(kwFloats(kw, "alpha"), size...)
	case "multinomial":
		return g.Multinomial(kwI(kw, "n"), kwFloats(kw, "pvals"), size...)
	}
	return nil, fmt.Errorf("unknown function %s", c.Fn)
}

func runLegacy(r *RandomState, c call) (*ndarray.Array, error) {
	kw, size := c.Kw, sizeOf(c.Size)
	switch c.Fn {
	case "rand":
		return r.Rand(size...)
	case "randn":
		return r.Randn(size...)
	case "random_sample":
		return r.RandomSample(size...)
	case "randint":
		dt := kwDType(kw, ndarray.Int64)
		lo, hi := bigBounds(kw)
		if dt == ndarray.Uint64 {
			return integersUint64(r.bg, true, lo.Uint64(), hi.Uint64(), false, size)
		}
		return r.RandintOf(dt, lo.Int64(), hi.Int64(), size...)
	case "normal":
		return r.Normal(kwF(kw, "loc", 0), kwF(kw, "scale", 1), size...)
	case "uniform":
		return r.Uniform(kwF(kw, "low", 0), kwF(kw, "high", 1), size...)
	case "standard_normal":
		return r.StandardNormal(size...)
	case "choice":
		if name, ok := kw["a"].(string); ok {
			return r.Choice(namedArray(name), choiceOpts(kw), size...)
		}
		return r.ChoiceN(kwI(kw, "a"), choiceOpts(kw), size...)
	case "shuffle":
		a := namedArray(kw["x"].(string))
		return a, r.Shuffle(a)
	case "permutation":
		if name, ok := kw["x"].(string); ok {
			return r.PermutationOf(namedArray(name))
		}
		return r.Permutation(int(kwI(kw, "x")))
	case "binomial":
		return r.Binomial(kwI(kw, "n"), kwF(kw, "p", 0), size...)
	case "poisson":
		return r.Poisson(kwF(kw, "lam", 1), size...)
	case "standard_exponential":
		return r.StandardExponential(size...)
	case "exponential":
		return r.Exponential(kwF(kw, "scale", 1), size...)
	case "standard_gamma":
		return r.StandardGamma(kwF(kw, "shape", 0), size...)
	case "gamma":
		return r.Gamma(kwF(kw, "shape", 0), kwF(kw, "scale", 1), size...)
	case "chisquare":
		return r.ChiSquare(kwF(kw, "df", 0), size...)
	case "lognormal":
		return r.Lognormal(kwF(kw, "mean", 0), kwF(kw, "sigma", 1), size...)
	case "multinomial":
		return r.Multinomial(kwI(kw, "n"), kwFloats(kw, "pvals"), size...)
	case "bytes":
		return bytesArray(r.Bytes(int(kwI(kw, "length")))), nil
	}
	return nil, fmt.Errorf("unknown function %s", c.Fn)
}

// compare returns how a result differs from NumPy's: a structural
// difference (dtype, shape, or a differing integer) as an error message,
// and for floats the number of differing values and the largest ULP
// distance.
func compare(got *ndarray.Array, want encoded, scale float64) (msg string, diff int, maxULP int64) {
	if got.DType().String() != want.DType {
		return fmt.Sprintf("dtype %v, NumPy %s", got.DType(), want.DType), 0, 0
	}
	if fmt.Sprint(got.Shape()) != fmt.Sprint(want.Shape) {
		return fmt.Sprintf("shape %v, NumPy %v", got.Shape(), want.Shape), 0, 0
	}
	vals := flatStrings(got)
	for i, w := range want.V {
		if vals[i] == w {
			continue
		}
		if !got.DType().IsFloat() {
			return fmt.Sprintf("[%d] = %s, NumPy %s", i, vals[i], w), 0, 0
		}
		gf, _ := strconv.ParseFloat(vals[i], 64)
		wf, _ := strconv.ParseFloat(w, 64)
		diff++
		u := ulps(gf, wf, got.DType() == ndarray.Float32)
		if scale > 0 {
			u = int64(math.Ceil(math.Abs(gf-wf) / (scale * 0x1p-52)))
		}
		maxULP = max(maxULP, u)
	}
	return "", diff, maxULP
}

func ulps(a, b float64, single bool) int64 {
	if single {
		d := int64(math.Float32bits(float32(a))) - int64(math.Float32bits(float32(b)))
		return max(d, -d)
	}
	d := int64(math.Float64bits(a)) - int64(math.Float64bits(b))
	return max(d, -d)
}

func flatStrings(a *ndarray.Array) []string {
	var out []string
	switch a.DType() {
	case ndarray.Float64:
		d, _ := ndarray.Data[float64](a)
		for _, v := range d {
			out = append(out, strconv.FormatFloat(v, 'g', -1, 64))
		}
	case ndarray.Float32:
		d, _ := ndarray.Data[float32](a)
		for _, v := range d {
			out = append(out, strconv.FormatFloat(float64(v), 'g', -1, 64))
		}
	case ndarray.Bool:
		d, _ := ndarray.Data[bool](a)
		for _, v := range d {
			out = append(out, map[bool]string{true: "1", false: "0"}[v])
		}
	case ndarray.Uint64:
		d, _ := ndarray.Data[uint64](a)
		for _, v := range d {
			out = append(out, strconv.FormatUint(v, 10))
		}
	default:
		d, _ := ndarray.Data[int64](a.AsType(ndarray.Int64))
		for _, v := range d {
			out = append(out, strconv.FormatInt(v, 10))
		}
	}
	return out
}

func normFloat(s string) string {
	f, _ := strconv.ParseFloat(s, 64)
	return strconv.FormatFloat(f, 'g', -1, 64)
}

type libmStat struct {
	calls, diffCalls, values, diffValues int
	maxULP                               int64
}

// replay runs every script of the fixture and returns, per sampler, how
// many float values differ from NumPy's; it fails the test on anything
// else that differs.
func replay(t *testing.T) map[string]*libmStat {
	stats := map[string]*libmStat{}
	for i, s := range fx.Cases {
		var g *Generator
		var r *RandomState
		if s.API == "Generator" {
			g = NewGenerator(newBG(s.BG, s.Seed))
		} else {
			r = legacySeeds[s.Seed]()
		}
		for j, c := range s.Calls {
			if c.R.DType == "float64" || c.R.DType == "float32" {
				for k := range c.R.V {
					c.R.V[k] = normFloat(c.R.V[k])
				}
			}
			var got *ndarray.Array
			var err error
			if g != nil {
				got, err = runGenerator(g, c)
			} else {
				got, err = runLegacy(r, c)
			}
			where := fmt.Sprintf("case %d call %d %s/%s/%s %s %v size %v", i, j, s.API, s.BG, s.Seed, c.Fn, c.Kw, c.Size)
			if err != nil {
				t.Errorf("%s: %v", where, err)
				break
			}
			msg, diff, u := compare(got, c.R, absULP[c.Fn])
			if msg != "" {
				t.Errorf("%s: %s", where, msg)
				break
			}
			st := stats[c.Fn]
			if st == nil {
				st = &libmStat{}
				stats[c.Fn] = st
			}
			st.calls++
			st.values += len(c.R.V)
			if diff == 0 {
				continue
			}
			if testing.Verbose() {
				t.Logf("%s: %d values differ, up to %d ULP", where, diff, u)
			}
			st.diffCalls++
			st.diffValues += diff
			st.maxULP = max(st.maxULP, u)
			if bound, ok := libmULP[c.Fn]; !ok || u > bound {
				t.Errorf("%s: %d values differ, up to %d ULP (bound %d)", where, diff, u, bound)
				break
			}
		}
	}
	return stats
}

// TestMatchesNumPy replays the fixture: everything to the bit, except the
// documented ULP-level differences of the C library functions.
func TestMatchesNumPy(t *testing.T) {
	stats := replay(t)
	names := make([]string, 0, len(stats))
	for k := range stats {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		st := stats[k]
		if st.diffValues > 0 {
			t.Logf("%-22s %4d/%4d calls, %5d/%6d values differ, max %d ULP", k, st.diffCalls, st.calls, st.diffValues, st.values, st.maxULP)
		}
	}
}

// leBytes returns the array's elements as little-endian bytes, as numpy's
// tobytes() of a little-endian array.
func leBytes(a *ndarray.Array) []byte {
	var b []byte
	switch a.DType() {
	case ndarray.Float64:
		d, _ := ndarray.Data[float64](a)
		for _, v := range d {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
		}
	case ndarray.Float32:
		d, _ := ndarray.Data[float32](a)
		for _, v := range d {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
	case ndarray.Int64:
		d, _ := ndarray.Data[int64](a)
		for _, v := range d {
			b = binary.LittleEndian.AppendUint64(b, uint64(v))
		}
	case ndarray.Int8:
		d, _ := ndarray.Data[int8](a)
		for _, v := range d {
			b = append(b, byte(v))
		}
	case ndarray.Uint16:
		d, _ := ndarray.Data[uint16](a)
		for _, v := range d {
			b = binary.LittleEndian.AppendUint16(b, v)
		}
	case ndarray.Bool:
		d, _ := ndarray.Data[bool](a)
		for _, v := range d {
			b = append(b, map[bool]byte{true: 1}[v])
		}
	}
	return b
}

// TestLongRunsMatchNumPy compares long runs, which reach the rare paths
// (ziggurat tails and wedges, rejection loops), by digest.
func TestLongRunsMatchNumPy(t *testing.T) {
	for _, d := range fx.Digests {
		c := call{Fn: d.Fn, Kw: d.Kw, Size: d.Size}
		var got *ndarray.Array
		var err error
		if d.API == "Generator" {
			got, err = runGenerator(NewGenerator(newBG("PCG64", d.Seed)), c)
		} else {
			got, err = runLegacy(legacySeeds[d.Seed](), c)
		}
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(leBytes(got))
		if got.DType().String() != d.D.DType || hex.EncodeToString(sum[:]) != d.D.SHA256 {
			t.Errorf("%s/%s %s %v size %v: %v digest differs from NumPy's", d.API, d.Seed, d.Fn, d.Kw, d.Size, got.DType())
		}
	}
}
