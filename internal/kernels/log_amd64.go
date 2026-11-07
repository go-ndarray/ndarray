package kernels

import "math"

// logConsts is the constant block logAVX2 reads (asmgen/amd64/gen.go,
// logKernel), each value as a 4-lane vector at the offset the kernel uses.
type logConsts struct {
	lo, hiMax, n1loM1, n1hi [4]uint64  // offsets 0, 32, 64, 96
	off, idxMask, topMask   [4]uint64  // 128, 160, 192
	magic                   [4]uint64  // 224
	magicSub                [4]float64 // 256
	x2048                   [4]uint64  // 288
	negOne, ln2hi, ln2lo    [4]float64 // 320, 352, 384
	a0, a1, a2, a3, a4      [4]float64 // 416, 448, 480, 512, 544
}

var logK = func() *logConsts {
	k := new(logConsts)
	for i := range 4 {
		k.lo[i], k.hiMax[i] = 0x0010<<48, 0x7fefffffffffffff
		k.n1loM1[i] = math.Float64bits(1.0-0x1p-4) - 1
		k.n1hi[i] = math.Float64bits(1.0 + 0x1.09p-4)
		k.off[i], k.idxMask[i], k.topMask[i] = logOff, logN-1, 0xfff<<52
		k.magic[i], k.magicSub[i], k.x2048[i] = 0x4330000000000000, 0x1p52+2048, 0x800
		k.negOne[i], k.ln2hi[i], k.ln2lo[i] = -1, logLn2hi, logLn2lo
		k.a0[i], k.a1[i], k.a2[i], k.a3[i], k.a4[i] = logA0, logA1, logA2, logA3, logA4
	}
	return k
}()

// useLogAVX2 selects the AVX2 kernel; a var so tests can force the Go path.
var useLogAVX2 = hasAVX2() && useFMA

// logFast runs blocks of 4 on the AVX2 kernel, which hands back each block
// holding an input Log does not take through its table path (not a positive
// normal, or within 2^-4 of 1) to logScalar, as is the tail. Both run the
// same operations in the same order, so they agree bit for bit.
func logFast(dst, src []float64) {
	i := 0
	if useLogAVX2 {
		n := len(src) &^ 3
		for i < n {
			i += logAVX2(&dst[i], &src[i], n-i, logK, &logTab[0])
			if i < n {
				logScalar(dst[i:i+4], src[i:i+4])
				i += 4
			}
		}
	}
	logScalar(dst[i:], src[i:])
}

//go:noescape
func logAVX2(dst, src *float64, n int, k *logConsts, tab *logEntry) (done int)
