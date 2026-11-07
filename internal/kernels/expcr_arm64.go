package kernels

// expCRNEONConsts is the constant block exp32CRNEON reads (asmgen/arm64/
// gen.go, exp32CRKernel) into V16-V26, then the table it indexes.
type expCRNEONConsts struct {
	shift, invLn2, ln2Hi, ln2Lo [2]float64 // offsets 0, 16, 32, 48
	c0, c1, c2, c3              [2]float64 // 64, 80, 96, 112
	idxMask, expMask            [2]uint64  // 128, 144
	lim                         [4]float32 // 160
	table                       [64]uint64 // 176
}

var expCRKN = func() *expCRNEONConsts {
	k := new(expCRNEONConsts)
	for i := range 2 {
		k.shift[i], k.invLn2[i], k.ln2Hi[i], k.ln2Lo[i] = crShift, crInvLn2, crLn2Hi, crLn2Lo
		k.c0[i], k.c1[i], k.c2[i], k.c3[i] = crC0, crC1, crC2, crC3
		k.idxMask[i], k.expMask[i] = 63, 0xfff0000000000000
	}
	k.lim = [4]float32{512, 512, 512, 512}
	k.table = crMant
	return k
}()

// exp32Fast is the correctly rounded float32 exp: blocks of 8 on the NEON
// kernel, which hands back each block holding a lane with |x| > 512 or a NaN
// to exp32CR, as is the tail. Both run expCR's operations in its order, so
// they agree bit for bit.
func exp32Fast(dst, src []float32) {
	src = src[:len(dst)]
	i, n := 0, len(dst)&^7
	for i < n {
		i += exp32CRNEON(&dst[i], &src[i], n-i, expCRKN)
		if i < n {
			exp32CR(dst[i:i+8], src[i:i+8])
			i += 8
		}
	}
	exp32CR(dst[i:], src[i:])
}

// log32Fast is the correctly rounded float32 log: logCR in Go here.
func log32Fast(dst, src []float32) { logCRSlice(dst, src) }

//go:noescape
func exp32CRNEON(dst, src *float32, n int, k *expCRNEONConsts) (done int)
