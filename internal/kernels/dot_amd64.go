package kernels

// useFMA selects the FMA dot kernel. It is a var so tests can force the pure-Go
// path on hardware that has FMA (never the reverse: that would fault).
var useFMA = hasFMA()

// dotSIMD is the inner product of two equal-length slices: the AVX2/FMA
// kernel when the CPU has FMA, else dotRange.
func dotSIMD(a, b []float64) float64 {
	if !useFMA || len(a) == 0 {
		return dotRange(a, b)
	}
	_ = b[len(a)-1] // the kernel reads len(a) of b
	return dotFMA(&a[0], &b[0], len(a))
}

//go:noescape
func dotFMA(a, b *float64, n int) float64

// hasFMA reports CPU and OS support for the FMA kernel (go-asmgen FeatureProbe).
func hasFMA() bool
