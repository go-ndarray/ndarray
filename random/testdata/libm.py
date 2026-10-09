# Generates random/testdata/libm.json, correctly rounded log, log1p, exp, expm1 and pow
# (mpmath, 300 bits) for libm_test.go:
#   uv run --with mpmath python random/testdata/libm.py > random/testdata/libm.json
import math, random, json, mpmath, struct
mpmath.mp.prec = 300
random.seed(7)
def f2s(x): return float.hex(x)
cases = {"log": [], "log1p": [], "exp": [], "expm1": [], "pow": []}
def rnd_double():
    return struct.unpack('<d', struct.pack('<Q', random.getrandbits(63)))[0]
for i in range(400):
    for x in [random.random(), random.uniform(0, 10), random.random()*1e-300, 1 + random.uniform(-1e-9, 1e-9),
              rnd_double()]:
        if x > 0 and math.isfinite(x):
            cases["log"].append([f2s(x), f2s(float(mpmath.log(x)))])
    for u in [-random.random(), random.uniform(-1e-3, 1e-3), random.uniform(-1e-12, 1e-12), random.uniform(0, 100), -0.5 + random.uniform(-0.1, 0.1)]:
        cases["log1p"].append([f2s(u), f2s(float(mpmath.log1p(u)))])
    for x in [random.uniform(-20, 5), random.uniform(-745, 709.7), random.uniform(-1e-6, 1e-6)]:
        cases["exp"].append([f2s(x), f2s(float(mpmath.exp(x)))])
    for x in [random.uniform(-3, 3), random.uniform(-1e-12, 1e-12), random.uniform(-1e-35, 1e-35), random.uniform(-50, 60)]:
        cases["expm1"].append([f2s(x), f2s(float(mpmath.expm1(x)))])
    for x, y in [(random.random(), random.uniform(0.5, 20)), (random.uniform(0, 10), random.uniform(-30, 30)),
                 (2.0, random.uniform(0, 1000)), (random.random(), random.uniform(1, 100))]:
        v = mpmath.mpf(x) ** mpmath.mpf(y)
        if v < mpmath.mpf(2)**-1022 or v > mpmath.mpf(2)**1023:
            continue
        cases["pow"].append([f2s(x), f2s(y), f2s(float(v))])
print(json.dumps(cases))
