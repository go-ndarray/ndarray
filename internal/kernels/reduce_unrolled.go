//go:build !amd64

package kernels

// maxUnrolled / minUnrolled are the four-accumulator NaN-propagating extreme
// reducers used by the non-amd64 SIMD dispatch (arm64 and the four scalar
// arches). amd64 has its own packed MAXPD/MINPD kernels (reduce_amd64.go ->
// sum_amd64.s), so these Go versions would be dead code there — hence the
// !amd64 build tag, which keeps every build's reachable statements covered
// (the amd64 coverage job validates the .s kernel instead; arm64/generic cover
// these). They keep FOUR independent builtin-max/min accumulators so the
// dependency chain is broken: the builtin max/min lowers to the hardware
// FMAXD/FMIND on arm64, and four parallel chains hide its latency (~3.6x over a
// single accumulator). max/min is associative for the NaN-propagating rule, so
// the result is bit-identical to the serial Max/Min oracle (any NaN in any lane
// -> NaN; same extreme otherwise). a is non-empty.

func maxUnrolled(a []float64) float64 {
	n := len(a)
	if n < 4 {
		return Max(a)
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	i := 4
	for ; i+4 <= n; i += 4 {
		m0 = max(m0, a[i])
		m1 = max(m1, a[i+1])
		m2 = max(m2, a[i+2])
		m3 = max(m3, a[i+3])
	}
	m := max(max(m0, m1), max(m2, m3))
	for ; i < n; i++ {
		m = max(m, a[i])
	}
	return m
}

func minUnrolled(a []float64) float64 {
	n := len(a)
	if n < 4 {
		return Min(a)
	}
	m0, m1, m2, m3 := a[0], a[1], a[2], a[3]
	i := 4
	for ; i+4 <= n; i += 4 {
		m0 = min(m0, a[i])
		m1 = min(m1, a[i+1])
		m2 = min(m2, a[i+2])
		m3 = min(m3, a[i+3])
	}
	m := min(min(m0, m1), min(m2, m3))
	for ; i < n; i++ {
		m = min(m, a[i])
	}
	return m
}
