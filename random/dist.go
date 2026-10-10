package random

import "math"

// The scalar samplers below are ports of numpy/random/src/distributions/
// distributions.c, statement for statement: the same draws in the same
// order, the same floating-point operations in the same order. A product
// that feeds a sum is converted explicitly (float64(a*b) + c): Go may fuse
// a*b+c into one FMA on arm64, ppc64le, s390x and riscv64, which rounds once
// instead of twice and changes the last bit. NumPy's x86-64 builds do not
// fuse; see docs/random.md for its aarch64 builds, which do.

func nextFloat(bg BitGenerator) float32 {
	return float32(bg.Uint32()>>8) * (1.0 / 16777216.0)
}

var (
	signDouble = [2]float64{1, -1}
	signFloat  = [2]float32{1, -1}
)

func standardExponential(bg BitGenerator) float64 {
	for {
		ri := bg.Uint64()
		ri >>= 3
		idx := uint8(ri)
		ri >>= 8
		x := float64(ri) * weDouble[idx]
		if ri < keDouble[idx] {
			return x
		}
		if idx == 0 {
			return zigExpR - log1p(-bg.Float64())
		}
		if float64((feDouble[idx-1]-feDouble[idx])*bg.Float64())+feDouble[idx] < exp(-x) {
			return x
		}
	}
}

func standardExponentialF(bg BitGenerator) float32 {
	for {
		ri := bg.Uint32()
		ri >>= 1
		idx := uint8(ri)
		ri >>= 8
		x := float32(ri) * weFloat[idx]
		if ri < keFloat[idx] {
			return x
		}
		if idx == 0 {
			return zigExpRF - log1pf(-nextFloat(bg))
		}
		if float32((feFloat[idx-1]-feFloat[idx])*nextFloat(bg))+feFloat[idx] < expf(-x) {
			return x
		}
	}
}

func standardNormal(bg BitGenerator) float64 {
	r := bg.Uint64()
	idx := r & 0xff
	r >>= 8
	rabs := (r >> 1) & 0x000fffffffffffff
	x := float64(rabs) * wiDouble[idx] * signDouble[r&1]
	if rabs < kiDouble[idx] {
		return x
	}
	return normalSlow(bg, idx, rabs, x)
}

// normalSlow finishes a ziggurat draw that missed the rectangle: the tail
// for layer 0, the wedge test for the others, and a fresh draw on rejection.
func normalSlow(bg BitGenerator, idx, rabs uint64, x float64) float64 {
	for {
		if idx == 0 {
			for {
				xx := float64(-zigNorInvR * log1p(-bg.Float64()))
				yy := -log1p(-bg.Float64())
				if yy+yy > float64(xx*xx) {
					if (rabs>>8)&1 != 0 {
						return -(zigNorR + xx)
					}
					return zigNorR + xx
				}
			}
		}
		if float64((fiDouble[idx-1]-fiDouble[idx])*bg.Float64())+fiDouble[idx] < exp(-0.5*x*x) {
			return x
		}
		r := bg.Uint64()
		idx = r & 0xff
		r >>= 8
		rabs = (r >> 1) & 0x000fffffffffffff
		x = float64(rabs) * wiDouble[idx] * signDouble[r&1]
		if rabs < kiDouble[idx] {
			return x
		}
	}
}

// standardNormalPCG is standardNormal's fast path on the default bit
// generator, called directly (no interface call per draw); the rare paths
// go through standardNormal's code.
func standardNormalPCG(p *PCG64) float64 {
	r := p.Uint64()
	idx := r & 0xff
	r >>= 8
	rabs := (r >> 1) & 0x000fffffffffffff
	x := float64(rabs) * wiDouble[idx] * signDouble[r&1]
	if rabs < kiDouble[idx] {
		return x
	}
	return normalSlow(p, idx, rabs, x)
}

func standardNormalF(bg BitGenerator) float32 {
	for {
		r := bg.Uint32()
		idx := r & 0xff
		rabs := (r >> 9) & 0x0007fffff
		x := float32(rabs) * wiFloat[idx] * signFloat[(r>>8)&1]
		if rabs < kiFloat[idx] {
			return x
		}
		if idx == 0 {
			for {
				xx := float32(-zigNorInvRF * log1pf(-nextFloat(bg)))
				yy := -log1pf(-nextFloat(bg))
				if yy+yy > float32(xx*xx) {
					if (rabs>>8)&1 != 0 {
						return -(zigNorRF + xx)
					}
					return zigNorRF + xx
				}
			}
		}
		// The C code evaluates exp(-0.5 * x * x) in double precision.
		xd := float64(x)
		if float64(float32(float32((fiFloat[idx-1]-fiFloat[idx])*nextFloat(bg))+fiFloat[idx])) < exp(-0.5*xd*xd) {
			return x
		}
	}
}

func standardGamma(bg BitGenerator, shape float64) float64 {
	switch {
	case shape == 1.0:
		return standardExponential(bg)
	case shape == 0.0:
		return 0.0
	case shape < 1.0:
		for {
			u := bg.Float64()
			v := standardExponential(bg)
			if u <= 1.0-shape {
				x := pow(u, 1./shape)
				if x <= v {
					return x
				}
			} else {
				y := -log((1 - u) / shape)
				x := pow(1.0-shape+float64(shape*y), 1./shape)
				if x <= v+y {
					return x
				}
			}
		}
	}
	b := shape - 1./3.
	c := 1. / math.Sqrt(9*b)
	for {
		var x, v float64
		for {
			x = standardNormal(bg)
			v = 1.0 + float64(c*x)
			if v > 0.0 {
				break
			}
		}
		v = float64(v * v * v)
		u := bg.Float64()
		if u < 1.0-float64(0.0331*(x*x)*(x*x)) {
			return float64(b * v)
		}
		if log(u) < float64(0.5*x*x)+float64(b*(1.-v+log(v))) {
			return float64(b * v)
		}
	}
}

func standardGammaF(bg BitGenerator, shape float32) float32 {
	switch {
	case shape == 1.0:
		return standardExponentialF(bg)
	case shape == 0.0:
		return 0.0
	case shape < 1.0:
		for {
			u := nextFloat(bg)
			v := standardExponentialF(bg)
			if u <= 1.0-shape {
				x := powf(u, 1.0/shape)
				if x <= v {
					return x
				}
			} else {
				y := -logf((1.0 - u) / shape)
				x := powf(1.0-shape+float32(shape*y), 1.0/shape)
				if x <= v+y {
					return x
				}
			}
		}
	}
	b := shape - 1.0/3.0
	c := 1.0 / float32(math.Sqrt(float64(9.0*b)))
	for {
		var x, v float32
		for {
			x = standardNormalF(bg)
			v = 1.0 + float32(c*x)
			if v > 0.0 {
				break
			}
		}
		v = float32(v * v * v)
		u := nextFloat(bg)
		if u < 1.0-float32(0.0331*(x*x)*(x*x)) {
			return float32(b * v)
		}
		if logf(u) < float32(0.5*x*x)+float32(b*(1.0-v+logf(v))) {
			return float32(b * v)
		}
	}
}

// loggam is numpy's random_loggam (SPECFUN's log-gamma).
func loggam(x float64) float64 {
	a := [10]float64{8.333333333333333e-02, -2.777777777777778e-03,
		7.936507936507937e-04, -5.952380952380952e-04,
		8.417508417508418e-04, -1.917526917526918e-03,
		6.410256410256410e-03, -2.955065359477124e-02,
		1.796443723688307e-01, -1.39243221690590e+00}
	if x == 1.0 || x == 2.0 {
		return 0.0
	}
	var n int64
	if x < 7.0 {
		n = int64(7 - x)
	}
	x0 := x + float64(n)
	x2 := (1.0 / x0) * (1.0 / x0)
	const lg2pi = 1.8378770664093453e+00
	gl0 := a[9]
	for k := 8; k >= 0; k-- {
		gl0 = float64(gl0*x2) + a[k]
	}
	gl := gl0/x0 + float64(0.5*lg2pi) + float64((x0-0.5)*log(x0)) - x0
	if x < 7.0 {
		for k := int64(1); k <= n; k++ {
			gl -= log(x0 - 1.0)
			x0 -= 1.0
		}
	}
	return gl
}

func normal(bg BitGenerator, loc, scale float64) float64 {
	return loc + float64(scale*standardNormal(bg))
}

func uniform(bg BitGenerator, lower, rng float64) float64 {
	return lower + float64(rng*bg.Float64())
}

const betaTinyThreshold = 3e-103

func beta(bg BitGenerator, a, b float64) float64 {
	if a <= 1.0 && b <= 1.0 {
		if a < betaTinyThreshold && b < betaTinyThreshold {
			u := bg.Float64()
			if (a+b)*u < a {
				return 1
			}
			return 0
		}
		for {
			u := bg.Float64()
			v := bg.Float64()
			x := pow(u, 1.0/a)
			y := pow(v, 1.0/b)
			xpy := x + y
			if xpy <= 1.0 && u+v > 0.0 {
				if x > 0 && y > 0 {
					return x / xpy
				}
				logX := log(u) / a
				logY := log(v) / b
				delta := logX - logY
				if delta > 0 {
					return exp(-log1p(exp(-delta)))
				}
				return exp(delta - log1p(exp(delta)))
			}
		}
	}
	ga := standardGamma(bg, a)
	gb := standardGamma(bg, b)
	return ga / (ga + gb)
}

func chisquare(bg BitGenerator, df float64) float64 {
	return float64(2.0 * standardGamma(bg, df/2.0))
}

func fDist(bg BitGenerator, dfnum, dfden float64) float64 {
	s1 := float64(chisquare(bg, dfnum) * dfden)
	s2 := float64(chisquare(bg, dfden) * dfnum)
	return s1 / s2
}

func standardCauchy(bg BitGenerator) float64 {
	a := standardNormal(bg)
	return a / standardNormal(bg)
}

func pareto(bg BitGenerator, a float64) float64 {
	return expm1(standardExponential(bg) / a)
}

func weibull(bg BitGenerator, a float64) float64 {
	if a == 0.0 {
		return 0.0
	}
	return pow(standardExponential(bg), 1./a)
}

func power(bg BitGenerator, a float64) float64 {
	return pow(-expm1(-standardExponential(bg)), 1./a)
}

func laplace(bg BitGenerator, loc, scale float64) float64 {
	for {
		u := bg.Float64()
		if u > 0.0 {
			lo := u + u
			hi := 2.0 - u - u
			s := 0
			if u >= 0.5 {
				s = 1
			}
			return loc + float64(signDouble[s]*scale*log(min(lo, hi)))
		}
	}
}

func gumbel(bg BitGenerator, loc, scale float64) float64 {
	for {
		u := 1.0 - bg.Float64()
		if u < 1.0 {
			return loc - float64(scale*log(-log(u)))
		}
	}
}

func logistic(bg BitGenerator, loc, scale float64) float64 {
	for {
		u := bg.Float64()
		if u > 0.0 {
			return loc + float64(scale*log(u/(1.0-u)))
		}
	}
}

func lognormal(bg BitGenerator, mean, sigma float64) float64 {
	return exp(normal(bg, mean, sigma))
}

func rayleigh(bg BitGenerator, mode float64) float64 {
	return float64(mode * math.Sqrt(2.0*standardExponential(bg)))
}

func standardT(bg BitGenerator, df float64) float64 {
	num := standardNormal(bg)
	denom := standardGamma(bg, df/2)
	return float64(math.Sqrt(df/2)*num) / math.Sqrt(denom)
}

func poissonMult(bg BitGenerator, lam float64) int64 {
	enlam := exp(-lam)
	var x int64
	prod := 1.0
	for {
		prod *= bg.Float64()
		if prod > enlam {
			x++
		} else {
			return x
		}
	}
}

const (
	ls2pi   = 0.91893853320467267
	twelfth = 0.083333333333333333333333
)

func poissonPtrs(bg BitGenerator, lam float64) int64 {
	slam := math.Sqrt(lam)
	loglam := log(lam)
	b := 0.931 + float64(2.53*slam)
	a := -0.059 + float64(0.02483*b)
	invalpha := 1.1239 + 1.1328/(b-3.4)
	vr := 0.9277 - 3.6224/(b-2)
	for {
		u := bg.Float64() - 0.5
		v := bg.Float64()
		us := 0.5 - math.Abs(u)
		k := int64(math.Floor(float64((2*a/us+b)*u) + lam + 0.43))
		if us >= 0.07 && v <= vr {
			return k
		}
		if k < 0 || (us < 0.013 && v > us) {
			continue
		}
		if log(v)+log(invalpha)-log(a/(us*us)+b) <=
			-lam+float64(float64(k)*loglam)-loggam(float64(k)+1) {
			return k
		}
	}
}

func poisson(bg BitGenerator, lam float64) int64 {
	if lam >= 10 {
		return poissonPtrs(bg, lam)
	}
	if lam == 0 {
		return 0
	}
	return poissonMult(bg, lam)
}

func negativeBinomial(bg BitGenerator, n, p float64) int64 {
	y := float64(standardGamma(bg, n) * ((1 - p) / p))
	return poisson(bg, y)
}

// binomialCache is numpy's binomial_t: the setup of the last (n, p), kept
// between calls by a Generator.
type binomialCache struct {
	has                                                 bool
	nsave                                               int64
	psave                                               float64
	m                                                   int64
	r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4 float64
}

// binomialBTPE is numpy's random_binomial_btpe; legacy selects the NumPy
// 1.16 version RandomState keeps (13680 for 13860 and all four error terms
// added).
func binomialBTPE(bg BitGenerator, n int64, p float64, bc *binomialCache, legacy bool) int64 {
	if !bc.has || bc.nsave != n || bc.psave != p {
		bc.nsave, bc.psave, bc.has = n, p, true
		bc.r = min(p, 1.0-p)
		bc.q = 1.0 - bc.r
		bc.fm = float64(float64(n)*bc.r) + bc.r
		bc.m = int64(math.Floor(bc.fm))
		bc.p1 = math.Floor(float64(2.195*math.Sqrt(float64(n)*bc.r*bc.q))-float64(4.6*bc.q)) + 0.5
		bc.xm = float64(bc.m) + 0.5
		bc.xl = bc.xm - bc.p1
		bc.xr = bc.xm + bc.p1
		bc.c = 0.134 + 20.5/(15.3+float64(bc.m))
		a := (bc.fm - bc.xl) / (bc.fm - float64(bc.xl*bc.r))
		bc.laml = float64(a * (1.0 + a/2.0))
		a = (bc.xr - bc.fm) / (bc.xr * bc.q)
		bc.lamr = float64(a * (1.0 + a/2.0))
		bc.p2 = float64(bc.p1 * (1.0 + float64(2.0*bc.c)))
		bc.p3 = bc.p2 + bc.c/bc.laml
		bc.p4 = bc.p3 + bc.c/bc.lamr
	}
	r, q, m, p1, xm, xl, xr, c := bc.r, bc.q, bc.m, bc.p1, bc.xm, bc.xl, bc.xr, bc.c
	laml, lamr, p2, p3, p4 := bc.laml, bc.lamr, bc.p2, bc.p3, bc.p4
	nf := float64(n)
	var y int64
	for {
		nrq := float64(nf * r * q)
		u := float64(bg.Float64() * p4)
		v := bg.Float64()
		if u <= p1 {
			y = int64(math.Floor(xm - float64(p1*v) + u))
			break
		}
		if u <= p2 {
			x := xl + (u-p1)/c
			v = float64(v*c) + 1.0 - math.Abs(float64(m)-x+0.5)/p1
			if v > 1.0 {
				continue
			}
			y = int64(math.Floor(x))
		} else if u <= p3 {
			y = int64(math.Floor(xl + log(v)/laml))
			if y < 0 || v == 0.0 {
				continue
			}
			v = float64(v * (u - p2) * laml)
		} else {
			y = int64(math.Floor(xr - log(v)/lamr))
			if y > n || v == 0.0 {
				continue
			}
			v = float64(v * (u - p3) * lamr)
		}
		k := y - m
		if k < 0 {
			k = -k
		}
		kf := float64(k)
		if k > 20 && kf < nrq/2.0-1 {
			rho := float64((kf / nrq) * ((float64(kf*(kf/3.0+0.625))+0.16666666666666666)/nrq + 0.5))
			t := float64(-kf*kf) / float64(2*nrq)
			A := log(v)
			if A < t-rho {
				break
			}
			if A > t+rho {
				continue
			}
			x1 := float64(y) + 1
			f1 := float64(m) + 1
			z := nf + 1 - float64(m)
			w := nf - float64(y) + 1
			if A > btpeBound(xm, nf, m, y, r, q, x1, f1, z, w, legacy) {
				continue
			}
			break
		}
		s := r / q
		a := float64(s * (nf + 1))
		F := 1.0
		if m < y {
			for i := m + 1; i <= y; i++ {
				F = float64(F * (a/float64(i) - s))
			}
		} else if m > y {
			for i := y + 1; i <= m; i++ {
				F /= a/float64(i) - s
			}
		}
		if v > F {
			continue
		}
		break
	}
	// numpy flips y when p > 0.5 here; binomial only passes p <= 0.5.
	return y
}

// btpeBound is the right-hand side of BTPE's final squeeze, with the
// Stirling error terms of f1, z, x1 and w.
func btpeBound(xm, nf float64, m, y int64, r, q, x1, f1, z, w float64, legacy bool) float64 {
	c0 := 13860.
	if legacy {
		c0 = 13680.
	}
	term := func(v float64) float64 {
		v2 := v * v
		return (c0 - (462.-(132.-(99.-140./v2)/v2)/v2)/v2) / v / 166320.
	}
	s := float64(xm*log(f1/x1)) + float64((nf-float64(m)+0.5)*log(z/w)) +
		float64(float64(y-m)*log(w*r/(x1*q))) + term(f1) + term(z)
	if legacy {
		return s + term(x1) + term(w)
	}
	return s - term(x1) - term(w)
}

// binomialInversion is numpy's random_binomial_inversion; legacy computes
// q**n as exp(n*log(q)) where the current code uses exp(n*log1p(-p)).
func binomialInversion(bg BitGenerator, n int64, p float64, bc *binomialCache, legacy bool) int64 {
	nf := float64(n)
	if !bc.has || bc.nsave != n || bc.psave != p {
		bc.nsave, bc.psave, bc.has = n, p, true
		bc.q = 1.0 - p
		if legacy {
			bc.r = exp(nf * log(bc.q))
		} else {
			bc.r = exp(nf * log1p(-p))
		}
		bc.c = float64(nf * p)
		bc.m = int64(min(nf, bc.c+float64(10.0*math.Sqrt(float64(bc.c*bc.q)+1))))
	}
	q, qn, bound := bc.q, bc.r, bc.m
	var x int64
	px := qn
	u := bg.Float64()
	for u > px {
		x++
		if x > bound {
			x = 0
			px = qn
			u = bg.Float64()
		} else {
			u -= px
			px = float64(float64(n-x+1)*p*px) / float64(float64(x)*q)
		}
	}
	return x
}

func binomial(bg BitGenerator, p float64, n int64, bc *binomialCache, legacy bool) int64 {
	if !legacy && (n == 0 || p == 0.0) {
		return 0
	}
	if p <= 0.5 {
		if p*float64(n) <= 30.0 {
			return binomialInversion(bg, n, p, bc, legacy)
		}
		return binomialBTPE(bg, n, p, bc, legacy)
	}
	q := 1.0 - p
	if q*float64(n) <= 30.0 {
		return n - binomialInversion(bg, n, q, bc, legacy)
	}
	return n - binomialBTPE(bg, n, q, bc, legacy)
}

func noncentralChisquare(bg BitGenerator, df, nonc float64) float64 {
	if math.IsNaN(nonc) {
		return math.NaN()
	}
	if nonc == 0 {
		return chisquare(bg, df)
	}
	if 1 < df {
		chi2 := chisquare(bg, df-1)
		n := standardNormal(bg) + math.Sqrt(nonc)
		return chi2 + float64(n*n)
	}
	i := poisson(bg, nonc/2.0)
	return chisquare(bg, df+float64(2*i))
}

func noncentralF(bg BitGenerator, dfnum, dfden, nonc float64) float64 {
	t := noncentralChisquare(bg, dfnum, nonc) * dfden
	return t / (chisquare(bg, dfden) * dfnum)
}

func wald(bg BitGenerator, mean, scale float64) float64 {
	y := standardNormal(bg)
	y = float64(mean * y * y)
	d := 1 + math.Sqrt(1+float64(4*scale)/y)
	x := float64(mean * (1 - 2/d))
	u := bg.Float64()
	if u <= mean/(mean+x) {
		return x
	}
	return float64(mean*mean) / x
}

func vonmises(bg BitGenerator, mu, kappa float64) float64 {
	if math.IsNaN(kappa) {
		return math.NaN()
	}
	if kappa < 1e-8 {
		return math.Pi * (2*bg.Float64() - 1)
	}
	var s float64
	if kappa < 1e-5 {
		s = 1./kappa + kappa
	} else if kappa <= 1e6 {
		r := 1 + math.Sqrt(1+float64(4*kappa*kappa))
		rho := (r - math.Sqrt(2*r)) / (2 * kappa)
		s = (1 + float64(rho*rho)) / (2 * rho)
	} else {
		result := mu + float64(math.Sqrt(1./kappa)*standardNormal(bg))
		if result < -math.Pi {
			result += 2 * math.Pi
		}
		if result > math.Pi {
			result -= 2 * math.Pi
		}
		return result
	}
	var w float64
	for {
		u := bg.Float64()
		z := cos(math.Pi * u)
		w = (1 + float64(s*z)) / (s + z)
		y := float64(kappa * (s - w))
		v := bg.Float64()
		if float64(y*(2-y))-v >= 0 || log(y/v)+1-y >= 0 {
			break
		}
	}
	u := bg.Float64()
	result := acos(w)
	if u < 0.5 {
		result = -result
	}
	result += mu
	neg := result < 0
	mod := math.Abs(result)
	mod = math.Mod(mod+math.Pi, 2*math.Pi) - math.Pi
	if neg {
		mod *= -1
	}
	return mod
}

func logseries(bg BitGenerator, p float64) int64 {
	r := log1p(-p)
	for {
		v := bg.Float64()
		if v >= p {
			return 1
		}
		u := bg.Float64()
		q := -expm1(float64(r * u))
		if v <= q*q {
			result := int64(math.Floor(1 + log(v)/log(q)))
			if result < 1 || v == 0.0 {
				continue
			}
			return result
		}
		if v >= q {
			return 1
		}
		return 2
	}
}

func geometricSearch(bg BitGenerator, p float64) int64 {
	x := int64(1)
	sum, prod := p, p
	q := 1.0 - p
	u := bg.Float64()
	for u > sum {
		prod = float64(prod * q)
		sum += prod
		x++
	}
	return x
}

func geometricInversion(bg BitGenerator, p float64) int64 {
	z := math.Ceil(-standardExponential(bg) / log1p(-p))
	if z >= 9.223372036854776e+18 {
		return math.MaxInt64
	}
	return int64(z)
}

func geometric(bg BitGenerator, p float64) int64 {
	if p >= 0.333333333333333333333333 {
		return geometricSearch(bg, p)
	}
	return geometricInversion(bg, p)
}

func zipf(bg BitGenerator, a float64) int64 {
	if a >= 1025 {
		return 1
	}
	am1 := a - 1.0
	b := pow(2.0, am1)
	umin := pow(float64(math.MaxInt64), -am1)
	for {
		u01 := bg.Float64()
		u := float64(u01*umin) + (1 - u01)
		v := bg.Float64()
		x := math.Floor(pow(u, -1.0/am1))
		// numpy rejects x beyond int64 or below 1 (neither can occur
		// short of pow rounding past 2^63) and tests the rest.
		if x <= float64(math.MaxInt64) && x >= 1.0 {
			t := pow(1.0+1.0/x, am1)
			if v*x*(t-1.0)/(b-1.0) <= t/b {
				return int64(x)
			}
		}
	}
}

func triangular(bg BitGenerator, left, mode, right float64) float64 {
	base := right - left
	leftbase := mode - left
	ratio := leftbase / base
	leftprod := float64(leftbase * base)
	rightprod := float64((right - mode) * base)
	u := bg.Float64()
	if u <= ratio {
		return left + math.Sqrt(float64(u*leftprod))
	}
	return right - math.Sqrt(float64((1.0-u)*rightprod))
}

// multinomial is numpy's random_multinomial: d-1 conditional binomials.
func multinomial(bg BitGenerator, n int64, mnix []int64, pix []float64, bc *binomialCache, legacy bool) {
	remaining := 1.0
	dn := n
	d := len(pix)
	for j := 0; j < d-1; j++ {
		mnix[j] = binomial(bg, pix[j]/remaining, dn, bc, legacy)
		dn -= mnix[j]
		if dn <= 0 {
			break
		}
		remaining -= pix[j]
	}
	if dn > 0 {
		mnix[d-1] = dn
	}
}
