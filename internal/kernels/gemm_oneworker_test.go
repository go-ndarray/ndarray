package kernels

import (
	"math/rand/v2"
	"runtime"
	"testing"
)

// TestGemmOneWorkerSameBits pins the one-worker path of matMul (#104): at
// GOMAXPROCS=1 a product above GemmThreshold takes packGemmRows instead of the
// banded parallel path, and on non-integer data the two must agree bit for
// bit (same tiles, same pc order), in float64 and float32, including m = 1,
// which takes the same branch at any GOMAXPROCS.
func TestGemmOneWorkerSameBits(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	for _, s := range [][3]int{{97, 131, 83}, {1, 300, 7000}, {256, 64, 200}} {
		m, k, n := s[0], s[1], s[2]
		if m*n < GemmThreshold {
			t.Fatalf("%v is below GemmThreshold %d: the case would not reach the branch", s, GemmThreshold)
		}
		a, b := make([]float64, m*k), make([]float64, k*n)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		for i := range b {
			b[i] = r.NormFloat64()
		}
		run := func(procs int) []float64 {
			defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
			dst := make([]float64, m*n)
			MatMulStridedP(dst, rowMajor(a, k), rowMajor(b, n), m, k, n)
			return dst
		}
		one, many := run(1), run(4)
		for i := range one {
			if one[i] != many[i] {
				t.Fatalf("float64 %v: element %d is %v on one worker, %v on four", s, i, one[i], many[i])
			}
		}
		if !HasGemm32() {
			continue
		}
		a32, b32 := make([]float32, m*k), make([]float32, k*n)
		for i := range a32 {
			a32[i] = float32(a[i])
		}
		for i := range b32 {
			b32[i] = float32(b[i])
		}
		run32 := func(procs int) []float32 {
			defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
			dst := make([]float32, m*n)
			MatMul32StridedP(dst, MatOf[float32]{Data: a32, RS: k, CS: 1}, MatOf[float32]{Data: b32, RS: n, CS: 1}, m, k, n)
			return dst
		}
		one32, many32 := run32(1), run32(4)
		for i := range one32 {
			if one32[i] != many32[i] {
				t.Fatalf("float32 %v: element %d is %v on one worker, %v on four", s, i, one32[i], many32[i])
			}
		}
	}
}
