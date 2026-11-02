package kernels

import "math"

// Correctly rounded float32 Exp and Log.
//
// float32 Exp and Log went through math.Exp and math.Log per element, 6-14x
// slower than NumPy's float32 SIMD on one Zen 3 core. Both now compute in
// float64 and round once, and both are proven correctly rounded over every
// float32 input (via64_exhaustive_test.go, expcr_test.go):
//
//   - Exp runs expCR (expcr.go), Arm's correctly rounded expf algorithm, on
//     an AVX2 kernel where there is one (expcr_amd64.go), in Go elsewhere;
//   - Log widens to this package's float64 Log (0.504 ULP) through a
//     256-element stack buffer, rounds once, and takes the 24 inputs whose
//     float64 result lies next to a float32 rounding midpoint from a table.
//
// Nothing is allocated. Log is called directly, not through a func value,
// which would make the buffer escape.

const via64Chunk = 256

// log32 writes the correctly rounded log(src[i]) into dst[i].
func log32(dst, src []float32) {
	var buf [via64Chunk]float64
	src = src[:len(dst)]
	for i := 0; i < len(dst); i += via64Chunk {
		b := buf[:min(via64Chunk, len(dst)-i)]
		s := src[i : i+len(b)]
		for j := range b {
			b[j] = float64(s[j])
		}
		Log(b, b)
		d := dst[i : i+len(b)]
		for j, v := range b {
			f := float32(v)
			// Within 8 float64 ULP of a float32 rounding midpoint (the 29
			// bits float32 drops are 0x10000000 +- 8): the double rounding
			// may be wrong, so take the table's answer.
			if math.Float64bits(v)&(1<<29-1)-(1<<28-8) < 16 {
				f = log32Exception(s[j], f)
			}
			d[j] = f
		}
	}
}

// log32Exceptions are the float32 inputs whose float64 log lies within 2
// float64 ULP of a float32 rounding midpoint, sorted, each with its correctly
// rounded log decided by a 300-bit reference. The exhaustive test lists them
// (NDARRAY_EXHAUSTIVE=1 NDARRAY_DUMP_HARD=1, TestLog32CorrectlyRounded): 24
// inputs, 5 of which rounded to the wrong float32 without this table. Log's
// float64 arithmetic is the same on every architecture (its multiply-add is
// math.FMA), so the list is too.
var log32Exceptions = [...][2]uint32{
	{0x0dc8bba4, 0xc289bac4}, {0x111c87f8, 0xc28085df}, {0x1a8446cb, 0xc24d0a45},
	{0x1f116ab8, 0xc233b53e}, {0x28e3fa26, 0xc1fa75e0}, {0x29e6126b, 0xc1ef4c02},
	{0x29fd22f8, 0xc1ee8859}, {0x2c4c24b7, 0xc1d48710}, {0x2e492984, 0xc1be76ee},
	{0x38dcbe38, 0xc1128ba4}, {0x3bf86ef0, 0xc09c399e}, {0x3c413d3a, 0xc08e158f},
	{0x41178feb, 0x400fe5e7}, {0x464d5b2b, 0x4117bcf1}, {0x4665a9a6, 0x41198725},
	{0x4c5d65a5, 0x418f034b}, {0x4d604ebe, 0x419a352c}, {0x4e85f412, 0x41a6b811},
	{0x5ee8984e, 0x422e4a21}, {0x65d890d3, 0x4254d1f9}, {0x66a8c860, 0x42595e46},
	{0x66abbd63, 0x4259700e}, {0x6f31a8ec, 0x42845a89}, {0x79e7ec37, 0x42a1ffb7},
}

// log32Exception returns the table's correctly rounded log of x if x is in
// log32Exceptions, and f otherwise.
func log32Exception(x, f float32) float32 {
	bits := math.Float32bits(x)
	lo, hi := 0, len(log32Exceptions)
	for lo < hi {
		m := (lo + hi) / 2
		switch e := log32Exceptions[m]; {
		case e[0] == bits:
			return math.Float32frombits(e[1])
		case e[0] < bits:
			lo = m + 1
		default:
			hi = m
		}
	}
	return f
}

// Exp32 writes the correctly rounded exp(src[i]) into dst[i].
func Exp32(dst, src []float32) { exp32Fast(dst, src) }

// Log32 writes the correctly rounded log(src[i]) into dst[i].
func Log32(dst, src []float32) { log32(dst, src) }

// Exp32P is Exp32, split across cores above ParThreshold.
func Exp32P(dst, src []float32) { split32(exp32Fast, dst, src) }

// Log32P is Log32, split across cores above ParThreshold.
func Log32P(dst, src []float32) { split32(log32, dst, src) }

func split32(f func(dst, src []float32), dst, src []float32) {
	n := len(dst)
	if n < ParThreshold {
		f(dst, src)
		return
	}
	parallelFor(n, numWorkers(n), func(lo, hi int) { f(dst[lo:hi], src[lo:hi]) })
}
