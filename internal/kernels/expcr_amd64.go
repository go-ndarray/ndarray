package kernels

// expCRConsts is the constant block exp32CRAVX2 reads (asmgen/amd64/gen.go,
// exp32CRKernel): each constant as a 4-lane vector at the offset the kernel
// uses, then the clamp bounds as 8 floats, then the 64 table mantissas.
type expCRConsts struct {
	shift, invLn2, ln2Hi, ln2Lo [4]float64 // offsets 0, 32, 64, 96
	c0, c1, c2, c3              [4]float64 // 128, 160, 192, 224
	idxMask, expMask            [4]uint64  // 256, 288
	lo, hi                      [8]float32 // 320, 352
	table                       [64]uint64 // 384
}

var expCRK = func() *expCRConsts {
	k := new(expCRConsts)
	for i := range 4 {
		k.shift[i], k.invLn2[i], k.ln2Hi[i], k.ln2Lo[i] = crShift, crInvLn2, crLn2Hi, crLn2Lo
		k.c0[i], k.c1[i], k.c2[i], k.c3[i] = crC0, crC1, crC2, crC3
		k.idxMask[i], k.expMask[i] = 63, 0xfff0000000000000
	}
	for i := range 8 {
		k.lo[i], k.hi[i] = -512, 512
	}
	k.table = crMant
	return k
}()

// useExp32AVX2 selects the AVX2 kernel; a var so tests can force the Go path.
var useExp32AVX2 = hasAVX2() && useFMA

// exp32Fast is the correctly rounded float32 exp: blocks of 8 on the AVX2
// kernel, the rest (and every element without AVX2+FMA) on exp32CR. Both
// compute expCR's operations in the same order, so they agree bit for bit.
func exp32Fast(dst, src []float32) {
	src = src[:len(dst)]
	n := 0
	if useExp32AVX2 {
		n = len(dst) &^ 7
		if n > 0 {
			exp32CRAVX2(&dst[0], &src[0], n, expCRK)
		}
	}
	exp32CR(dst[n:], src[n:])
}

// hasAVX2 reports CPU and OS support for AVX2 (go-asmgen FeatureProbe).
func hasAVX2() bool

//go:noescape
func exp32CRAVX2(dst, src *float32, n int, k *expCRConsts)
