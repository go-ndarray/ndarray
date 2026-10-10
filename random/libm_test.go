package random

import (
	_ "embed"
	"encoding/json"
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
)

// libm.json holds correctly rounded values of log, log1p, exp, expm1 and pow
// (mpmath at 300 bits) on random and edge arguments.
//
//go:embed testdata/libm.json
var libmJSON []byte

func hexF(t *testing.T, s string) float64 {
	t.Helper()
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestCorrectlyRounded(t *testing.T) {
	var c map[string][][]string
	if err := json.Unmarshal(libmJSON, &c); err != nil {
		t.Fatal(err)
	}
	one := map[string]func(float64) float64{"log": log, "log1p": log1p, "exp": exp, "expm1": expm1}
	for name, rows := range c {
		bad := 0
		for _, r := range rows {
			var got, want float64
			if name == "pow" {
				got, want = pow(hexF(t, r[0]), hexF(t, r[1])), hexF(t, r[2])
			} else {
				got, want = one[name](hexF(t, r[0])), hexF(t, r[1])
			}
			if got != want {
				if bad < 3 {
					t.Errorf("%s(%v) = %v, correctly rounded %v", name, r[:len(r)-1], got, want)
				}
				bad++
			}
		}
		t.Logf("%s: %d arguments, %d not correctly rounded", name, len(rows), bad)
	}
}

// TestFastPathsAgreeWithDoubleDouble checks the fast paths of log, exp and
// pow (about 2^-68, used when Ziv's rounding test passes) against the
// double-double evaluations, which libm.json checks against mpmath.
func TestFastPathsAgreeWithDoubleDouble(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	n := 400_000
	if testing.Short() {
		n = 40_000
	}
	var slow [3]int
	for i := range n {
		var x float64
		switch i % 4 {
		case 0:
			x = r.Float64()
		case 1:
			x = math.Float64frombits(r.Uint64() >> 1) // any positive double
		case 2:
			x = 1 + (r.Float64()-0.5)*1e-6
		default:
			x = r.Float64() * 100
		}
		if x > 0 && !math.IsInf(x, 0) && log(x) != logDD(x).round() {
			t.Fatalf("log(%v) = %v, double-double %v", x, log(x), logDD(x).round())
		}
		if _, _, ok := logFast(x); !ok {
			slow[0]++
		}
		y := (r.Float64() - 0.5) * 1400
		if i%3 == 0 {
			y = (r.Float64() - 0.5) * 1e-3
		}
		q, e := expDD(dd{y, 0})
		if want := scale(q, e); math.Abs(y) < 709 && exp(y) != want {
			t.Fatalf("exp(%v) = %v, double-double %v", y, exp(y), want)
		}
		px, py := r.Float64()*10, (r.Float64()-0.5)*60
		l := logDD(px)
		p, pe := twoProd(l.hi, py)
		q, e = expDD(dd{p, pe + l.lo*py})
		if want := scale(q, e); pow(px, py) != want {
			t.Fatalf("pow(%v, %v) = %v, double-double %v", px, py, pow(px, py), want)
		}
	}
	// The slow paths: subnormal arguments and results.
	if log(0x1p-1060) != logDD(0x1p-1060).round() || exp(-720) == 0 || pow(0x1p-1060, 0.5) != 0x1p-530 {
		t.Error("subnormal paths")
	}
}
