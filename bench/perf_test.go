package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with NumPy (docs/perf.md).
//
//	cd bench
//	GOMAXPROCS=1 go test -run TestPerf -cpu 1 .                CI mode: allocations, table, denominator
//	GOMAXPROCS=1 go test -run TestPerf -cpu 1 -perf.run .      measure (refuses at a 1-minute load of 8 or more)
//	GOMAXPROCS=1 go test -run TestPerf -perf.record .          allocation counts only (a deferred report)
//	go test -run 'TestDataParity|TestRefSmoke' -perf.parity .  same data on both sides; every NumPy call runs (needs uv)
//
// GOMAXPROCS=1 must be in the environment: ndarray/fft rests on go-fft/fft,
// which reads GOMAXPROCS once at init (go-fft/fft#107), so the GOMAXPROCS=1 that
// refbench sets around a run comes too late for it.

func allRows() []row {
	var rs []row
	rs = append(rs, arrayRows()...)
	rs = append(rs, ctorRows()...)
	rs = append(rs, fftRows()...)
	rs = append(rs, linalgRows()...)
	rs = append(rs, genRows()...)
	rs = append(rs, legacyRows()...)
	rs = append(rs, packageRows()...)
	return rs
}

func setupOf(r row) string {
	s := "["
	for i, d := range r.args {
		if i > 0 {
			s += ", "
		}
		s += d.py
	}
	return s + "]"
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range allRows() {
		arrs := make([]*ndarray.Array, len(r.args))
		for i, d := range r.args {
			arrs[i] = d.arr()
		}
		run := r.body(arrs)
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(setupOf(r), r.call, 1)
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: r.ref,
			Parity:    refbench.ParityNone, // value parity is each package's own NumPy fixtures
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					run()
				}
			},
			Ref: ref,
		})
	}
	return out
}

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

var parityFlag = flag.Bool("perf.parity", false, "TestDataParity, TestRefSmoke: start the NumPy worker through uv")

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		if os.Getenv("GOMAXPROCS") != "1" {
			t.Fatal("GOMAXPROCS=1 must be in the environment (go-fft/fft reads it once at init)")
		}
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-ndarray/ndarray"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

// TestDataParity checks, for every row with arguments, that the Go values and
// NumPy's are the same bytes: a ratio between two computations on different
// data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the NumPy worker through uv)")
	}
	w := startWorker(t)
	seen := map[string]bool{}
	for _, r := range allRows() {
		if len(r.args) == 0 {
			continue
		}
		setup := setupOf(r)
		if seen[setup] {
			continue
		}
		seen[setup] = true
		h := sha256.New()
		for _, d := range r.args {
			for _, v := range d.data {
				var b [8]byte
				binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
				h.Write(b[:])
			}
		}
		got, err := w.DataHash(setup, 1)
		if err != nil {
			t.Errorf("%s: %v", r.name, err)
			continue
		}
		if want := hex.EncodeToString(h.Sum(nil)); got != want {
			t.Errorf("%s: NumPy data %s, Go data %s", r.name, got, want)
		}
	}
	t.Logf("%d distinct argument sets compared", len(seen))
}

// TestRefSmoke runs every NumPy call once: a call that raises would otherwise
// show only in a timed run.
func TestRefSmoke(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the NumPy worker through uv)")
	}
	w := startWorker(t)
	rows := allRows()
	for _, r := range rows {
		if _, err := w.Timer(setupOf(r), r.call, 1)(1); err != nil {
			t.Errorf("%s: %v\n  call: %s", r.name, err, r.call)
		}
	}
	fmt.Println("rows:", len(rows))
}
