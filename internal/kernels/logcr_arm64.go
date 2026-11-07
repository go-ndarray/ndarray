package kernels

// logCRNEONConsts is the constant block log32CRNEON loads into V16-V28
// (asmgen/arm64/gen.go, log32CRKernel), then the tables it indexes.
type logCRNEONConsts struct {
	one, ln2, b0, b1, b2, w               [2]float64  // offsets 0, 16, 32, 48, 64, 80
	mmask, jadd, bias, sub, maxv, oneBits [4]uint32   // 96, 112, 128, 144, 160, 176
	expOr                                 [2]uint64   // 192
	tr, tl                                [65]float64 // 208, 728
}

var logCRKN = func() *logCRNEONConsts {
	k := new(logCRNEONConsts)
	for i := range 2 {
		k.one[i], k.ln2[i], k.w[i] = 1, 0x1.62e42fefa39efp-1, 0x1.f06p-33
		k.b0[i], k.b1[i], k.b2[i] = logCRB[0], logCRB[1], logCRB[2]
		k.expOr[i] = 1023 << 23
	}
	for i := range 4 {
		k.mmask[i], k.jadd[i], k.bias[i] = 1<<23-1, 1<<16, 127
		// The positive-normal test: ux + sub (wrapping) <= maxv as int32.
		k.sub[i], k.maxv[i], k.oneBits[i] = 0x7f800000, 0xfeffffff, 0x3f800000
	}
	k.tr, k.tl = logCRTr, logCRTl
	return k
}()

// log32Fast is the correctly rounded float32 log: blocks of 8 on the NEON
// kernel, which hands back each block it cannot finish (an input that is not
// a positive normal float32, or bounds that round apart) to logCR, as is the
// tail. Both run logCR's operations in its order, so they agree bit for bit.
func log32Fast(dst, src []float32) {
	src = src[:len(dst)]
	i, n := 0, len(dst)&^7
	for i < n {
		i += log32CRNEON(&dst[i], &src[i], n-i, logCRKN)
		if i < n {
			logCRSlice(dst[i:i+8], src[i:i+8])
			i += 8
		}
	}
	logCRSlice(dst[i:], src[i:])
}

//go:noescape
func log32CRNEON(dst, src *float32, n int, k *logCRNEONConsts) (done int)
