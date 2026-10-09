package random

import "math"

// The elementary functions numpy takes from the C library.
//
// log, log1p, exp, expm1 and pow are correctly rounded here: each is
// evaluated in double-double arithmetic (about 100 good bits, from exact
// FMA-based products) and rounded once. The C libraries numpy is built
// against (glibc, Apple's libm, the Windows CRT) round these within about
// half an ULP, so they too return the correctly rounded value almost always
// — measured on 20,000 arguments per function, Apple's x86-64 libm does in
// all but 0.06% (log) to 0.3% (exp) of them, where Go's math package
// differs from it in 6.5% (log) to 71% (pow). The result is a stream that
// matches numpy's to the bit in all but those rare cases, and that is the
// same on every Go architecture (math.Log, math.Exp ... use assembly on some
// and pure Go on others).
//
// The double-double result is within about 2^-100 of the true value, so it
// rounds correctly unless the true value lies within that distance of a
// midpoint between two doubles: about one argument in 2^47.
//
// cos and acos (only vonmises uses them) come from the math package; the C
// libraries are not correctly rounded there either (Apple's differs from the
// correctly rounded cos for 17% of arguments in [0, pi]).

// Go may fuse a product with a later sum into one FMA, even across
// statements and inlined calls, which breaks these error-free
// transformations; every product below is converted explicitly, which
// forbids it.

// twoSum returns s, e with s = fl(a+b) and s+e = a+b exactly.
func twoSum(a, b float64) (float64, float64) {
	s := a + b
	bb := s - a
	return s, (a - (s - bb)) + (b - bb)
}

// fastTwoSum is twoSum for |a| >= |b|.
func fastTwoSum(a, b float64) (float64, float64) {
	s := a + b
	return s, b - (s - a)
}

// twoProd returns p, e with p = fl(a*b) and p+e = a*b exactly.
func twoProd(a, b float64) (float64, float64) {
	p := float64(a * b) // the conversion forbids fusing p into a later sum
	return p, math.FMA(a, b, -p)
}

// dd is a double-double: the unevaluated sum hi + lo, |lo| <= ulp(hi)/2.
type dd struct{ hi, lo float64 }

func (a dd) add(b dd) dd {
	s, e := twoSum(a.hi, b.hi)
	e += a.lo + b.lo
	h, l := fastTwoSum(s, e)
	return dd{h, l}
}

func (a dd) mul(b dd) dd {
	p, e := twoProd(a.hi, b.hi)
	e += float64(a.hi*b.lo) + float64(a.lo*b.hi)
	h, l := fastTwoSum(p, e)
	return dd{h, l}
}

func (a dd) mulF(b float64) dd {
	p, e := twoProd(a.hi, b)
	e += float64(a.lo * b)
	h, l := fastTwoSum(p, e)
	return dd{h, l}
}

func (a dd) div(b dd) dd {
	q1 := a.hi / b.hi
	r := a.add(b.mulF(-q1))
	q2 := r.hi / b.hi
	r = r.add(b.mulF(-q2))
	q3 := r.hi / b.hi
	h, l := fastTwoSum(q1, q2)
	return dd{h, l}.add(dd{q3, 0})
}

func (a dd) round() float64 { return a.hi + a.lo }

// atanh2 returns 2*atanh(s) = log((1+s)/(1-s)) for |s| <= 2^-6, to about
// 2^-104 relative: 2s(1 + s²/3 + s⁴/5 + ...), the leading terms in
// double-double.
func atanh2(s dd) dd {
	z := s.mul(s)
	zf := z.hi
	tail := horner(zf, 1.0/11, 1.0/13, 1.0/15, 1.0/17, 1.0/19)
	p := invOdd[9].add(dd{float64(zf * tail), 0})
	for _, c := range [...]int{7, 5, 3} {
		p = z.mul(p).add(invOdd[c])
	}
	p = z.mul(p).add(dd{1, 0})
	return s.mul(p).mulF(2)
}

// ddRecip returns 1/c as a double-double.
func ddRecip(c float64) dd { return dd{1, 0}.div(dd{c, 0}) }

// invOdd[k] is 1/k for the odd k of atanh2; invFact[k] is 1/k!.
var (
	invOdd  = map[int]dd{3: ddRecip(3), 5: ddRecip(5), 7: ddRecip(7), 9: ddRecip(9)}
	invFact = [7]dd{{1, 0}, {1, 0}, {0.5, 0}, ddRecip(6), ddRecip(24), ddRecip(120), ddRecip(720)}
)

// horner evaluates c[0] + x*(c[1] + x*(...)) in plain double arithmetic,
// each product rounded on its own (no FMA), so that it is the same on
// every architecture.
func horner(x float64, c ...float64) float64 {
	r := c[len(c)-1]
	for i := len(c) - 2; i >= 0; i-- {
		r = c[i] + float64(x*r)
	}
	return r
}

// logDD returns log(x) for a positive finite x as a double-double.
func logDD(x float64) dd {
	m, e := math.Frexp(x) // x = m * 2^e, m in [0.5, 1)
	m *= 2
	e--
	if m >= math.Sqrt2 {
		m /= 2
		e++
	}
	j := int(math.Floor(m*64 + 0.5))
	c := float64(j) / 64
	num := m - c // exact: c/2 <= m <= 2c
	den := dd{m, 0}.add(dd{c, 0})
	s := dd{num, 0}.div(den)
	r := atanh2(s).add(dd{logTab[j-45][0], logTab[j-45][1]})
	if e != 0 {
		ef := float64(e)
		h, l := twoProd(ef, ln2Hi)
		r = dd{h, l + float64(ef*ln2Lo)}.add(r)
	}
	return r
}

// logFast returns log(x) for a positive finite x as hi + lo, to about
// 2^-70 relative, or ok = false for a subnormal x. Callers apply Ziv's
// rounding test and fall back on the slower logDD when it fails.
func logFast(x float64) (hi, lo float64, ok bool) {
	b := math.Float64bits(x)
	if b < 1<<52 {
		return 0, 0, false // subnormal: logDD normalises it
	}
	e := int(b>>52) - 1023
	m := math.Float64frombits(b&(1<<52-1) | 1023<<52) // in [1, 2)
	if m >= math.Sqrt2 {
		m *= 0.5
		e++
	}
	j := int(m*64 + 0.5)
	c := float64(j) / 64
	num := m - c // exact
	dh, dl := twoSum(m, c)
	sh := num / dh
	sl := (math.FMA(-sh, dh, num) - float64(sh*dl)) / dh
	z := float64(sh * sh)
	t := float64(sh * float64(z*horner(z, 2.0/3, 2.0/5, 2.0/7, 2.0/9)))
	ef := float64(e)
	ah, al := twoProd(ef, ln2Hi)
	h1, l1 := twoSum(ah, logTab[j-45][0])
	h2, l2 := twoSum(h1, 2*sh)
	low := l1 + l2 + al + float64(ef*ln2Lo) + logTab[j-45][1] + 2*sl + t
	hi, lo = fastTwoSum(h2, low)
	return hi, lo, true
}

// rounds reports whether hi+lo, known to within eps*|hi|, rounds to the
// same double at both ends of that interval, and returns that double.
func rounds(hi, lo, eps float64) (float64, bool) {
	d := math.Abs(hi) * eps
	r := hi + (lo - d)
	return r, r == hi+(lo+d)
}

func log(x float64) float64 {
	switch {
	case x > 0 && !math.IsInf(x, 1):
		if h, l, ok := logFast(x); ok {
			if r, ok := rounds(h, l, 0x1p-68); ok {
				return r
			}
		}
		return logDD(x).round()
	case x == 0:
		return math.Inf(-1)
	}
	return math.Log(x) // NaN, negative, +Inf
}

// log1pDD returns log(1+u) for u > -1 as a double-double.
func log1pDD(u float64) dd {
	if math.Abs(u) < 0x1p-5 {
		// s = u/(2+u) <= 2^-6 and log1p(u) = 2 atanh(s).
		return atanh2(dd{u, 0}.div(dd{2, 0}.add(dd{u, 0})))
	}
	h, l := twoSum(1, u)
	return logDD(h).add(dd{l / h, 0})
}

func log1p(u float64) float64 {
	switch {
	case u > -1 && !math.IsInf(u, 1):
		if u == 0 {
			return u // keeps -0
		}
		return log1pDD(u).round()
	case u == -1:
		return math.Inf(-1)
	}
	return math.Log1p(u) // NaN, < -1, +Inf
}

// expDD returns exp(a) for a double-double argument within the finite range,
// as a double-double and a power of two to scale it by.
func expDD(a dd) (dd, int) {
	k := math.Floor(float64(a.hi*(64/math.Ln2)) + 0.5)
	// r = a - k*ln2/64, with ln2/64 in three parts.
	p, pe := twoProd(k, ln2by64A)
	r := a.add(dd{-p, -pe}).add(dd{float64(-k * ln2by64B), float64(-k * ln2by64C)})
	// exp(r), |r| <= ln2/128: Taylor to degree 11, the first six terms
	// in double-double.
	tail := horner(r.hi, 1.0/5040, 1.0/40320, 1.0/362880, 1.0/3628800, 1.0/39916800)
	q := dd{tail, 0}
	for c := 6; c >= 0; c-- {
		q = r.mul(q).add(invFact[c])
	}
	ki := int(k)
	j := ki & 63
	q = q.mul(dd{exp2Tab[j][0], exp2Tab[j][1]})
	return q, (ki - j) / 64
}

func exp(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 709.8:
		return math.Inf(1)
	case x < -745.2:
		return 0
	}
	if h, l, e := expFast(x, 0); e > -1000 {
		if r, ok := rounds(h, l, 0x1p-66); ok {
			return math.Ldexp(r, e)
		}
	}
	q, e := expDD(dd{x, 0})
	return scale(q, e)
}

// expFast returns exp(xh+xl) as (hi+lo) * 2^e, to about 2^-67 relative,
// for |xh| <= 745.2 and |xl| <= 2^-40 |xh|. Callers apply Ziv's rounding
// test (and scale only normal results) and fall back on expDD.
func expFast(xh, xl float64) (hi, lo float64, e int) {
	k := math.Floor(float64(xh*(64/math.Ln2)) + 0.5)
	ph, pl := twoProd(k, ln2by64A)
	rh, rl := twoSum(xh, -ph)
	rh, rl = fastTwoSum(rh, rl+(xl-pl-float64(k*ln2by64B)))
	// exp(rh+rl) = 1 + rh + rh²(1/2 + rh/6 + ...) + rl(1 + rh), |rh| <= 2^-7.5.
	p := float64(float64(rh*rh) * horner(rh, 1.0/2, 1.0/6, 1.0/24, 1.0/120, 1.0/720, 1.0/5040))
	s1, e1 := fastTwoSum(1, rh)
	eh, el := fastTwoSum(s1, e1+rl+float64(rl*rh)+p)
	ki := int(k)
	j := ki & 63
	th, tl := exp2Tab[j][0], exp2Tab[j][1]
	hi, lo = twoProd(th, eh)
	lo += float64(th*el) + float64(tl*eh)
	hi, lo = fastTwoSum(hi, lo)
	return hi, lo, (ki - j) / 64
}

// scale returns q * 2^e rounded once when the result is normal; a subnormal
// result is rounded twice (to 53 bits, then to its precision).
func scale(q dd, e int) float64 {
	return math.Ldexp(q.round(), e)
}

func expm1(x float64) float64 {
	switch {
	case math.IsNaN(x) || x == 0:
		return x
	case x > 709.8:
		return math.Inf(1)
	case x < -40:
		return -1 // exp(x) < 2^-57: -1 + exp(x) rounds to -1
	case x > 700:
		return exp(x) // exp(x) > 2^1009: the -1 is far below an ULP
	case math.Abs(x) < 0x1p-30:
		// x + x²/2 + x³/6 + x⁴/24, the square exact.
		h, l := twoProd(x, x)
		t := dd{h / 2, l / 2}.add(dd{float64(x*(h/6)) + float64(h*h/24), 0})
		return dd{x, 0}.add(t).round()
	}
	q, e := expDD(dd{x, 0})
	q = q.mul(dd{math.Ldexp(1, e), 0})
	return q.add(dd{-1, 0}).round()
}

func pow(x, y float64) float64 {
	if !(x > 0) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) || y == 0 || x == 1 {
		return math.Pow(x, y) // the C99 special cases, and x <= 0
	}
	if lh, ll, ok := logFast(x); ok {
		ph, pl := twoProd(lh, y)
		pl += float64(ll * y)
		if math.Abs(ph) <= 709 {
			// The argument is good to 709*2^-70 absolute, so the result to
			// about 2^-60 relative.
			if h, l, e := expFast(ph, pl); e > -1000 {
				if r, ok := rounds(h, l, 0x1p-59); ok {
					return math.Ldexp(r, e)
				}
			}
		}
	}
	l := logDD(x)
	p, pe := twoProd(l.hi, y)
	a := dd{p, pe + float64(l.lo*y)}
	switch {
	case a.hi > 709.8:
		return math.Inf(1)
	case a.hi < -745.2:
		return 0
	}
	q, e := expDD(a)
	return scale(q, e)
}

func cos(x float64) float64  { return math.Cos(x) }
func acos(x float64) float64 { return math.Acos(x) }

func logf(x float32) float32    { return float32(log(float64(x))) }
func expf(x float32) float32    { return float32(exp(float64(x))) }
func log1pf(x float32) float32  { return float32(log1p(float64(x))) }
func powf(x, y float32) float32 { return float32(pow(float64(x), float64(y))) }
