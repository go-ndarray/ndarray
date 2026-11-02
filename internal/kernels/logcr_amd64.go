package kernels

// logCRConsts is the constant block log32CRAVX2 reads (asmgen/amd64/gen.go,
// log32CRKernel), at the offsets the kernel uses.
type logCRConsts struct {
	one, ln2, b0, b1, b2, w               [4]float64  // offsets 0, 32, 64, 96, 128, 160
	sub, maxv, mmask, jadd, bias, oneBits [8]uint32   // 192, 224, 256, 288, 320, 352
	expOr                                 [4]uint64   // 384
	tr, tl                                [65]float64 // 416, 936
}

var logCRK = func() *logCRConsts {
	k := new(logCRConsts)
	for i := range 4 {
		k.one[i], k.ln2[i], k.w[i] = 1, 0x1.62e42fefa39efp-1, 0x1.f06p-33
		k.b0[i], k.b1[i], k.b2[i] = logCRB[0], logCRB[1], logCRB[2]
		k.expOr[i] = 1023 << 23
	}
	for i := range 8 {
		k.sub[i], k.maxv[i], k.mmask[i] = 1<<23, 0x7effffff, 1<<23-1
		k.jadd[i], k.bias[i], k.oneBits[i] = 1<<16, 127, 0x3f800000
	}
	k.tr, k.tl = logCRTr, logCRTl
	return k
}()

// useLog32AVX2 selects the AVX2 kernel; a var so tests can force the Go path.
var useLog32AVX2 = hasAVX2()

// log32Fast is the correctly rounded float32 log: blocks of 8 on the AVX2
// kernel, which hands back each block it cannot finish (an input that is not
// a positive normal float32, or bounds that round apart, about 1 lane in
// 2800) to logCR, as is the tail. Both run logCR's operations in its order,
// so they agree bit for bit.
func log32Fast(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	if useLog32AVX2 {
		n := len(dst) &^ 7
		for i < n {
			i += log32CRAVX2(&dst[i], &src[i], n-i, logCRK)
			if i < n {
				logCRSlice(dst[i:i+8], src[i:i+8])
				i += 8
			}
		}
	}
	logCRSlice(dst[i:], src[i:])
}

//go:noescape
func log32CRAVX2(dst, src *float32, n int, k *logCRConsts) (done int)
