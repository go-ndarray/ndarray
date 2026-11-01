# Generates random/testdata/scalar.json, NumPy's output for the scalar draws
# of the random package (Float64, FillUniform, IntN, FillIntN, ShuffleInt64)
# on 48 seeds, for Generator over PCG64 and MT19937 and for RandomState over
# MT19937 (legacy seeding) and PCG64. Run by the Go test harness
# (TestRegenerateScalarFixture, GONDARRAY_REGEN=1), which starts an x86-64
# NumPy 2.5.3 -- the build that follows distributions.c operation by
# operation (aarch64 builds fuse a*b+c in uniform, see docs/random.md):
#
#   python scalar.py scalar.json
#
# Every case is a script: a fresh generator, then calls in order, each with
# its result, or null when NumPy raised. The calls are chosen per seed by a
# private random.Random so that PCG64's buffered half-word is exercised by
# every ordering of 32-bit and 64-bit draws.
import json, random, sys, platform
import numpy as np
from numpy.random import SeedSequence, PCG64, MT19937, Generator, RandomState

NSEEDS = 48
HIGHS = [1, 2, 3, 5, 7, 10, 100, 255, 256, 257, 65535, 65536, 10**6,
         2**31 - 1, 2**31, 2**31 + 1, 2**32 - 2, 2**32 - 1, 2**32, 2**32 + 1,
         3 * 2**30, 2**33, 2**40 + 7, 2**62 + 1, 2**63 - 2, 2**63 - 1]
BADHIGHS = [0, -1, -2**63]
UNIFORMS = [(0.0, 1.0), (-1.0, 1.0), (5.0, 5.0), (-3.5, 10.25), (0.1, 0.7),
            (1e-300, 1e300), (-1e300, 1e300), (0.0, 5e-324), (-0.0, 0.0),
            (3.0, 2.0), (1.0, -1.0), (float("-inf"), 0.0), (0.0, float("inf")),
            (float("nan"), 1.0), (0.0, float("nan")), (-1.7e308, 1.7e308),
            (1.7e308, -1.7e308), (float("inf"), float("inf"))]
SHUFFLES = [0, 1, 2, 3, 5, 8, 13, 50, 257]


def mk(api, bg, seed):
    if api == "gen":
        b = PCG64(seed) if bg == "PCG64" else MT19937(seed)
        return Generator(b)
    if bg == "MT19937":
        return RandomState(seed)  # legacy seeding, seed < 2**32
    return RandomState(PCG64(seed))


def fl(x):
    return repr(float(x))


def run(g, api, op, a):
    """Returns the encoded result, or None when NumPy raises."""
    try:
        if op == "f":
            return [fl(g.random() if api == "gen" else g.random_sample())]
        if op == "u":
            low, high, n = a
            return [fl(v) for v in g.uniform(low, high, size=n)]
        if op == "i":
            (high,) = a
            v = g.integers(0, high) if api == "gen" else g.randint(0, high)
            return [str(int(v))]
        if op == "fi":
            high, n = a
            v = (g.integers(0, high, size=n) if api == "gen"
                 else g.randint(0, high, size=n))
            return [str(int(z)) for z in v]
        if op == "sh":
            (n,) = a
            x = np.arange(n, dtype=np.int64)
            g.shuffle(x)
            return [str(int(z)) for z in x]
    except (ValueError, OverflowError):
        return None
    raise AssertionError(op)


def script(api, bg, seed):
    rnd = random.Random(7919 * seed + (1 if api == "gen" else 2))
    g = mk(api, bg, seed)
    calls = []
    for _ in range(36):
        k = rnd.random()
        if k < 0.18:
            op, a = "f", []
        elif k < 0.34:
            low, high = rnd.choice(UNIFORMS)
            op, a = "u", [low, high, rnd.choice([0, 1, 1, 2, 5, 9])]
        elif k < 0.60:
            op, a = "i", [rnd.choice(HIGHS + HIGHS + BADHIGHS)]
        elif k < 0.85:
            op, a = "fi", [rnd.choice(HIGHS + BADHIGHS), rnd.choice([0, 1, 2, 3, 6, 11])]
        else:
            op, a = "sh", [rnd.choice(SHUFFLES)]
        r = run(g, api, op, a)
        calls.append({"op": op, "a": [fl(v) if isinstance(v, float) else str(v) for v in a], "r": r})
    # An error must not consume anything: the next draws prove it.
    calls.append({"op": "f", "a": [], "r": run(g, api, "f", [])})
    calls.append({"op": "i", "a": ["1000003"], "r": run(g, api, "i", [1000003])})
    return {"api": api, "bg": bg, "seed": str(seed), "calls": calls}


def main():
    cases = []
    for seed in range(NSEEDS):
        cases.append(script("gen", "PCG64", seed))
        cases.append(script("gen", "MT19937", seed))
        cases.append(script("legacy", "MT19937", seed))
        cases.append(script("legacy", "PCG64", seed))
    out = {"numpy": np.__version__, "machine": platform.machine(), "cases": cases}
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, separators=(",", ":"))


main()
