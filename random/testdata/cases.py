# Generates random/testdata/cases.json, NumPy's output for every function of
# the package on several seeds and bit generators:
#
#   uv run --python cpython-3.12-macos-x86_64 --with numpy==2.5.3 \
#       python random/testdata/cases.py > random/testdata/cases.json
#
# The reference is an x86-64 build of NumPy (on Apple silicon it runs under
# Rosetta). NumPy's aarch64 builds let the C compiler fuse a*b+c into one FMA
# in some samplers (uniform, for one), so their last bits differ from its own
# x86-64 builds; the x86-64 output is the one that follows the C source
# operation by operation, and the one this package reproduces. See
# docs/random.md.
#
# Every case is a script: a fresh generator, then calls in order, each with
# its result. Integers are decimal strings, floats repr strings (exact).
import json, platform
import numpy as np
from numpy.random import (SeedSequence, PCG64, PCG64DXSM, MT19937, Philox,
                          SFC64, Generator, RandomState)

SEEDS = {
    "0": 0, "1": 1, "42": 42, "u64max": 2**64 - 1,
    "list": [1, 2, 3], "big": 2**128 + 12345,
}
BG = {"PCG64": PCG64, "PCG64DXSM": PCG64DXSM, "MT19937": MT19937,
      "Philox": Philox, "SFC64": SFC64}


def enc(v):
    v = np.asarray(v)
    out = []
    for z in v.ravel():
        if v.dtype.kind == 'f':
            out.append(repr(float(z)))
        elif v.dtype.kind == 'b':
            out.append("1" if z else "0")
        else:
            out.append(str(int(z)))
    return {"dtype": str(v.dtype), "shape": list(v.shape), "v": out}


ARANGE12 = np.arange(12).reshape(3, 4)
FLOATS6 = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


def gen_calls():
    c = []
    a = c.append
    a(("random", {}, 7)); a(("random", {"dtype": "float32"}, [2, 3])); a(("random", {}, None))
    for dt, bounds in [
        ("int8", [(0, 10, False), (-128, 127, True), (-5, 5, False), (3, 4, False), (0, 64, False)]),
        ("int16", [(0, 10, False), (-32768, 32767, True), (-300, 300, False), (0, 256, False)]),
        ("int32", [(0, 10, False), (-2**31, 2**31 - 1, True), (-70000, 70000, False), (0, 2**16, False)]),
        ("int64", [(0, 10, False), (-2**63, 2**63 - 1, True), (0, 2**32, False), (0, 2**32, True),
                   (-2**40, 2**40, False), (5, 2**31 + 5, False), (7, 8, False), (0, 3 * 2**30, False)]),
        ("uint8", [(0, 10, False), (0, 255, True), (200, 256, False)]),
        ("uint16", [(0, 10, False), (0, 65535, True), (1000, 1001, False)]),
        ("uint32", [(0, 10, False), (0, 2**32 - 1, True), (2**31, 2**32, False)]),
        ("uint64", [(0, 10, False), (0, 2**64 - 1, True), (2**63, 2**64 - 1, False), (0, 2**33, False),
                    (0, 3 * 2**62, False)]),
        ("bool", [(0, 2, False), (0, 1, True), (1, 2, False), (0, 1, False)]),
    ]:
        for lo, hi, ep in bounds:
            a(("integers", {"low": lo, "high": hi, "dtype": dt, "endpoint": ep}, 9))
    a(("integers", {"low": 0, "high": 100}, None))
    for dt in ["float64", "float32"]:
        a(("standard_normal", {"dtype": dt}, 60))
        a(("standard_exponential", {"dtype": dt}, 60))
        a(("standard_exponential", {"dtype": dt, "method": "inv"}, 10))
        for sh in [0.5, 1.0, 3.0, 0.0, 0.05]:
            a(("standard_gamma", {"shape": sh, "dtype": dt}, 20))
    a(("normal", {"loc": 1.5, "scale": 2.0}, 20))
    a(("exponential", {"scale": 2.0}, 10))
    a(("uniform", {"low": -1.0, "high": 3.0}, 20))
    a(("gamma", {"shape": 2.5, "scale": 1.5}, 10))
    for ab in [(0.5, 0.5), (2.0, 3.0), (1e-104, 1e-104), (0.01, 0.01), (1e-5, 1e-5)]:
        a(("beta", {"a": ab[0], "b": ab[1]}, 10))
    a(("chisquare", {"df": 3.0}, 10))
    a(("f", {"dfnum": 3.0, "dfden": 5.0}, 10))
    for df, nonc in [(3.0, 2.0), (0.5, 2.0), (3.0, 0.0)]:
        a(("noncentral_chisquare", {"df": df, "nonc": nonc}, 10))
    a(("noncentral_f", {"dfnum": 3.0, "dfden": 5.0, "nonc": 2.0}, 10))
    a(("standard_cauchy", {}, 10))
    a(("standard_t", {"df": 4.0}, 10))
    for mu, kappa in [(0.5, 2.0), (0.0, 1e-9), (0.0, 1e-6), (0.3, 1e7), (3.1415, 1e7), (-3.1415, 1e7)]:
        a(("vonmises", {"mu": mu, "kappa": kappa}, 10))
    a(("pareto", {"a": 3.0}, 10))
    a(("weibull", {"a": 2.0}, 10)); a(("weibull", {"a": 0.0}, 3))
    a(("power", {"a": 2.0}, 10))
    a(("laplace", {"loc": 0.5, "scale": 2.0}, 10))
    a(("gumbel", {"loc": 1.0, "scale": 2.0}, 10))
    a(("logistic", {"loc": 1.0, "scale": 2.0}, 10))
    a(("lognormal", {"mean": 0.5, "sigma": 0.25}, 10))
    a(("rayleigh", {"scale": 2.0}, 10))
    a(("wald", {"mean": 1.0, "scale": 2.0}, 10))
    a(("triangular", {"left": 0.0, "mode": 1.0, "right": 3.0}, 10))
    for n, p in [(10, 0.3), (1000, 0.4), (1000, 0.7), (5, 0.9), (0, 0.5), (10, 0.0), (100000, 0.5), (3, 1.0)]:
        a(("binomial", {"n": n, "p": p}, 20))
    a(("negative_binomial", {"n": 5.0, "p": 0.3}, 10))
    for lam in [3.0, 50.0, 0.0, 1e6, 10.0]:
        a(("poisson", {"lam": lam}, 20))
    a(("zipf", {"a": 2.0}, 10)); a(("zipf", {"a": 1.5}, 10)); a(("zipf", {"a": 2000.0}, 2))
    a(("geometric", {"p": 0.5}, 10)); a(("geometric", {"p": 0.05}, 10)); a(("geometric", {"p": 1e-300}, 2))
    a(("logseries", {"p": 0.5}, 10)); a(("logseries", {"p": 0.99}, 10))
    a(("choice", {"a": 10}, 5))
    a(("choice", {"a": 10}, None))
    a(("choice", {"a": 10, "replace": False}, 5))
    a(("choice", {"a": 10, "replace": False, "shuffle": False}, 5))
    a(("choice", {"a": 10000, "replace": False}, 5))
    a(("choice", {"a": 20000, "replace": False}, 500))
    a(("choice", {"a": 20000, "replace": False, "shuffle": False}, 1500))
    a(("choice", {"a": 5, "p": [0.1, 0.2, 0.3, 0.15, 0.25]}, 6))
    a(("choice", {"a": 5, "p": [0.1, 0.2, 0.3, 0.15, 0.25], "replace": False}, 4))
    a(("choice", {"a": "arange12", "axis": 1}, 2))
    a(("choice", {"a": "arange12"}, None))
    a(("choice", {"a": "floats6"}, [2, 2]))
    a(("shuffle", {"x": "arange10"}, None))
    a(("shuffle", {"x": "arange12", "axis": 0}, None))
    a(("shuffle", {"x": "arange12", "axis": 1}, None))
    a(("permutation", {"x": 10}, None))
    a(("permutation", {"x": "arange12", "axis": 1}, None))
    a(("permutation", {"x": "floats6"}, None))
    a(("permuted", {"x": "arange12"}, None))
    a(("permuted", {"x": "arange12", "axis": 1}, None))
    a(("permuted", {"x": "arange12", "axis": 0}, None))
    a(("bytes", {"length": 10}, None))
    a(("bytes", {"length": 0}, None))
    a(("dirichlet", {"alpha": [0.5, 1.0, 2.0]}, 3))
    a(("dirichlet", {"alpha": [0.05, 0.02, 0.01]}, 2))
    a(("dirichlet", {"alpha": [0.05, 0.0, 0.0]}, 2))
    a(("multinomial", {"n": 20, "pvals": [0.2, 0.3, 0.5]}, 3))
    a(("multinomial", {"n": 1000, "pvals": [0.1, 0.2, 0.7]}, None))
    a(("multinomial", {"n": 10, "pvals": [1.0, 0.0, 0.0]}, 2))
    a(("multinomial", {"n": 0, "pvals": [0.5, 0.5]}, 2))
    return c


def arg_array(v):
    return {"arange10": np.arange(10), "arange12": ARANGE12.copy(), "floats6": FLOATS6.copy()}.get(v, v) \
        if isinstance(v, str) else v


def run_gen(g, fn, kw, size):
    kw = {k: arg_array(v) for k, v in kw.items()}
    if fn in ("shuffle",):
        x = kw.pop("x")
        g.shuffle(x, **kw)
        return x
    if fn in ("permutation", "permuted"):
        x = kw.pop("x")
        return getattr(g, fn)(x, **kw)
    if fn == "bytes":
        return np.frombuffer(g.bytes(kw["length"]), dtype=np.uint8)
    if fn == "dirichlet":
        return g.dirichlet(kw["alpha"], size=size)
    if fn == "multinomial":
        return g.multinomial(kw["n"], kw["pvals"], size=size)
    if fn == "choice":
        a = kw.pop("a")
        return g.choice(a, size=size, **kw)
    return getattr(g, fn)(size=size, **kw)


def make_bg(kind, seed):
    return BG[kind](SEEDS[seed])


cases = []


def script(api, kind, seed, calls):
    if api == "Generator":
        g = Generator(make_bg(kind, seed))
        run = run_gen
    else:
        g = RandomState(seed)
        run = run_legacy
    out = []
    for fn, kw, size in calls:
        r = run(g, fn, dict(kw), size)
        out.append({"fn": fn, "kw": kw, "size": size, "r": enc(r)})
    cases.append({"api": api, "bg": kind, "seed": seed, "calls": out})


GEN_CALLS = gen_calls()
for seed in SEEDS:
    for call in GEN_CALLS:
        script("Generator", "PCG64", seed, [call])
SHORT = [c for c in GEN_CALLS if c[0] in ("random", "integers", "standard_normal", "standard_exponential",
                                          "standard_gamma", "binomial", "choice", "shuffle", "bytes")]
for kind in ["PCG64DXSM", "MT19937", "Philox", "SFC64"]:
    for seed in ["0", "42", "big"]:
        for call in SHORT:
            script("Generator", kind, seed, [call])
# The buffered upper half of a 64-bit draw outlives a call: interleave.
MIXED = [("integers", {"low": 0, "high": 10, "dtype": "int32"}, 3), ("random", {}, 2),
         ("integers", {"low": 0, "high": 10, "dtype": "int16"}, 1),
         ("standard_normal", {"dtype": "float32"}, 3), ("random", {"dtype": "float32"}, 1),
         ("integers", {"low": 0, "high": 10, "dtype": "uint8"}, 5), ("random", {}, 1),
         ("bytes", {"length": 3}, None), ("standard_normal", {}, 2)]
for kind in BG:
    for seed in SEEDS:
        script("Generator", kind, seed, MIXED)

# Legacy RandomState.
LSEEDS = {"0": 0, "1": 1, "42": 42, "u32max": 2**32 - 1, "arr": [1, 2, 3], "arr700": list(range(700))}


def run_legacy(r, fn, kw, size):
    kw = {k: arg_array(v) for k, v in kw.items()}
    if fn == "shuffle":
        x = kw.pop("x")
        r.shuffle(x)
        return x
    if fn == "permutation":
        return r.permutation(kw["x"])
    if fn == "bytes":
        return np.frombuffer(r.bytes(kw["length"]), dtype=np.uint8)
    if fn in ("rand", "randn"):
        return getattr(r, fn)(*(size or []))
    if fn == "choice":
        a = kw.pop("a")
        return r.choice(a, size=size, **kw)
    if fn == "multinomial":
        return r.multinomial(kw["n"], kw["pvals"], size=size)
    return getattr(r, fn)(size=size, **kw)


LEG_CALLS = [("rand", {}, [3, 2]), ("rand", {}, None), ("randn", {}, [7]), ("random_sample", {}, 5),
             ("randint", {"low": 0, "high": 10}, 8), ("randint", {"low": 0, "high": 2**40}, 5),
             ("randint", {"low": -2**63, "high": 2**63 - 1}, 5)]
for dt, lo, hi in [("int8", -5, 5), ("int16", -300, 300), ("int32", 0, 2**31 - 1), ("uint8", 0, 256),
                   ("uint16", 7, 8), ("uint32", 0, 2**32), ("uint64", 0, 2**64 - 1), ("bool", 0, 2),
                   ("int64", 0, 2**32)]:
    LEG_CALLS.append(("randint", {"low": lo, "high": hi, "dtype": dt}, 8))
LEG_CALLS += [
    ("normal", {"loc": 1.0, "scale": 2.0}, 6), ("uniform", {"low": -1.0, "high": 3.0}, 6),
    ("uniform", {"low": 3.0, "high": -1.0}, 4), ("standard_normal", {}, 5),
    ("choice", {"a": 10}, 5), ("choice", {"a": 10, "replace": False}, 5),
    ("choice", {"a": 5, "p": [0.1, 0.2, 0.3, 0.15, 0.25]}, 4),
    ("choice", {"a": 5, "p": [0.1, 0.2, 0.3, 0.15, 0.25], "replace": False}, 3),
    ("choice", {"a": "floats6"}, [2, 2]), ("choice", {"a": 10}, None),
    ("shuffle", {"x": "arange10"}, None), ("shuffle", {"x": "arange12"}, None),
    ("permutation", {"x": 10}, None), ("permutation", {"x": "arange12"}, None),
    ("permutation", {"x": "floats6"}, None),
    ("binomial", {"n": 10, "p": 0.3}, 10), ("binomial", {"n": 1000, "p": 0.4}, 10),
    ("binomial", {"n": 1000, "p": 0.7}, 10), ("binomial", {"n": 0, "p": 0.5}, 3),
    ("binomial", {"n": 100000, "p": 0.5}, 10),
    ("poisson", {"lam": 3.0}, 10), ("poisson", {"lam": 50.0}, 10), ("poisson", {"lam": 0.0}, 2),
    ("standard_exponential", {}, 5), ("exponential", {"scale": 2.0}, 5),
    ("standard_gamma", {"shape": 0.5}, 5), ("standard_gamma", {"shape": 3.0}, 5),
    ("standard_gamma", {"shape": 1.0}, 3), ("standard_gamma", {"shape": 0.0}, 2),
    ("gamma", {"shape": 2.0, "scale": 2.0}, 5), ("chisquare", {"df": 3.0}, 5),
    ("lognormal", {"mean": 0.5, "sigma": 0.25}, 5),
    ("multinomial", {"n": 20, "pvals": [0.2, 0.3, 0.5]}, 3),
    ("bytes", {"length": 10}, None),
]
for seed in LSEEDS:
    for call in LEG_CALLS:
        r = RandomState(LSEEDS[seed])
        out = [{"fn": call[0], "kw": call[1], "size": call[2], "r": enc(run_legacy(r, call[0], dict(call[1]), call[2]))}]
        cases.append({"api": "RandomState", "bg": "MT19937", "seed": seed, "calls": out})
    # The polar method caches its second value across calls.
    r = RandomState(LSEEDS[seed])
    calls = [("randn", {}, [3]), ("rand", {}, [1]), ("randn", {}, [1]), ("normal", {"loc": 0.0, "scale": 1.0}, 2),
             ("standard_gamma", {"shape": 3.0}, 3), ("randn", {}, [1])]
    cases.append({"api": "RandomState", "bg": "MT19937", "seed": seed,
                  "calls": [{"fn": f, "kw": k, "size": s, "r": enc(run_legacy(r, f, dict(k), s))} for f, k, s in calls]})

# Samplers that call the C library on every draw, long enough to meet its
# rare misroundings (see libmULP in numpy_test.go).
for seed in ["0", "42"]:
    for fn, kw, size in [("randn", {}, [3000]), ("normal", {"loc": 1.0, "scale": 3.0}, 2000),
                         ("standard_gamma", {"shape": 3.0}, 2000)]:
        r = RandomState(LSEEDS[seed])
        cases.append({"api": "RandomState", "bg": "MT19937", "seed": seed,
                      "calls": [{"fn": fn, "kw": kw, "size": size, "r": enc(run_legacy(r, fn, dict(kw), size))}]})
    script("Generator", "PCG64", seed, [("standard_gamma", {"shape": 0.5}, 3000)])

# Bit generators: raw streams, seeding, jumps.
bitgens = []
for kind, cls in BG.items():
    for seed, s in SEEDS.items():
        bg = cls(s)
        st = bg.state
        rec = {"bg": kind, "seed": seed, "raw": [str(int(v)) for v in bg.random_raw(700 if kind == "MT19937" else 100)]}
        if kind in ("PCG64", "PCG64DXSM"):
            rec["state"] = [str(st["state"]["state"]), str(st["state"]["inc"])]
        elif kind == "MT19937":
            rec["state"] = [str(int(st["state"]["key"][i])) for i in (0, 1, 2, 623)] + [str(st["state"]["pos"])]
        elif kind == "Philox":
            rec["state"] = [str(int(v)) for v in st["state"]["key"]]
        else:
            rec["state"] = [str(int(v)) for v in st["state"]["state"]]
        for j in ((1, 3) if kind != "SFC64" else ()):
            rec[f"jumped{j}"] = [str(int(v)) for v in cls(s).jumped(j).random_raw(5)]
        if kind in ("PCG64", "PCG64DXSM", "Philox"):
            b = cls(s)
            b.random_raw(3)
            rec["advance"] = [str(int(v)) for v in b.advance(2**100 + 7).random_raw(5)]
            b = cls(s)
            b.random_raw(7)
            rec["advance_neg"] = [str(int(v)) for v in b.advance(-3).random_raw(5)]
        kids = cls(s).spawn(2)
        rec["spawn"] = [[str(int(v)) for v in k.random_raw(5)] for k in kids]
        bitgens.append(rec)

ph = Philox(counter=np.array([1, 2, 3, 2**64 - 1], dtype=np.uint64), key=np.array([5, 6], dtype=np.uint64))
philox_key = {"raw": [str(int(v)) for v in ph.random_raw(12)]}
ph = Philox(counter=np.array([2**64 - 1, 2**64 - 1, 0, 0], dtype=np.uint64), key=np.array([1, 2], dtype=np.uint64))
philox_key["carry"] = [str(int(v)) for v in ph.advance(1).random_raw(4)]
legacy_mt = []
for seed, s in LSEEDS.items():
    m = MT19937()
    m._legacy_seeding(s)
    legacy_mt.append({"seed": seed, "raw": [str(int(v)) for v in m.random_raw(700)],
                      "jumped": [str(int(v)) for v in m.jumped(1).random_raw(5)]})
    m = MT19937()
    m._legacy_seeding(s)
    legacy_mt[-1]["jumped_fresh"] = [str(int(v)) for v in m.jumped(1).random_raw(5)]

seedseqs = []
for name, kw in [("0", {"entropy": 0}), ("42", {"entropy": 42}), ("u64max", {"entropy": 2**64 - 1}),
                 ("list", {"entropy": [1, 2, 3]}), ("big", {"entropy": 2**128 + 12345}),
                 ("long", {"entropy": list(range(1, 11))}), ("empty", {"entropy": []}),
                 ("pool8", {"entropy": 7, "pool_size": 8}),
                 ("spawnkey", {"entropy": 7, "spawn_key": (3, 2**40)}),
                 ("spawnkey_long", {"entropy": list(range(6)), "spawn_key": (1,)})]:
    ss = SeedSequence(**kw)
    kids = ss.spawn(3)
    more = ss.spawn(1)
    seedseqs.append({
        "name": name, "pool": [str(int(v)) for v in ss.pool],
        "state32": [str(int(v)) for v in ss.generate_state(9)],
        "state64": [str(int(v)) for v in ss.generate_state(5, np.uint64)],
        "kids": [{"key": [str(k) for k in c.spawn_key], "pool": [str(int(v)) for v in c.pool]} for c in kids + more],
    })

# Long runs, as SHA-256 digests of the little-endian bytes: they reach the
# rare paths (ziggurat tails and wedges, rejection loops) that short runs miss.
import hashlib
DIGESTS = [
    ("random", {}, 100000), ("random", {"dtype": "float32"}, 100000),
    ("standard_normal", {}, 200000), ("standard_normal", {"dtype": "float32"}, 200000),
    ("standard_exponential", {}, 200000), ("standard_exponential", {"dtype": "float32"}, 200000),
    ("integers", {"low": 0, "high": 1000}, 100000), ("integers", {"low": -100, "high": 100, "dtype": "int8"}, 100000),
    ("integers", {"low": 0, "high": 40000, "dtype": "uint16"}, 100000), ("integers", {"low": 0, "high": 2, "dtype": "bool"}, 100000),
    ("standard_gamma", {"shape": 3.0}, 50000),
    ("standard_gamma", {"shape": 3.0, "dtype": "float32"}, 50000),
    ("binomial", {"n": 1000, "p": 0.4}, 50000), ("binomial", {"n": 20, "p": 0.3}, 50000),
    ("poisson", {"lam": 50.0}, 50000), ("poisson", {"lam": 3.0}, 50000),
    ("normal", {"loc": 1.0, "scale": 3.0}, 100000), ("uniform", {"low": -2.0, "high": 5.0}, 100000),
    ("choice", {"a": 1000}, 50000), ("choice", {"a": 100000, "replace": False}, 5000),
]
LDIGESTS = [("rand", {}, [100000]), ("randint", {"low": 0, "high": 1000}, 100000),
            ("binomial", {"n": 1000, "p": 0.4}, 20000), ("poisson", {"lam": 50.0}, 20000),
            ("uniform", {"low": -1.0, "high": 2.0}, 50000)]


def digest(v):
    v = np.ascontiguousarray(v)
    b = v.astype(v.dtype.newbyteorder("<")).tobytes()
    return {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": hashlib.sha256(b).hexdigest()}


digests = []
for seed in ["0", "42"]:
    for fn, kw, size in DIGESTS:
        g = Generator(PCG64(SEEDS[seed]))
        digests.append({"api": "Generator", "seed": seed, "fn": fn, "kw": kw, "size": size,
                        "d": digest(run_gen(g, fn, dict(kw), size))})
    for fn, kw, size in LDIGESTS:
        r = RandomState(LSEEDS[seed])
        digests.append({"api": "RandomState", "seed": seed, "fn": fn, "kw": kw, "size": size,
                        "d": digest(run_legacy(r, fn, dict(kw), size))})

print(json.dumps({"numpy": np.__version__, "machine": platform.machine(), "cases": cases, "digests": digests,
                  "bitgens": bitgens, "philox_key": philox_key, "legacy_mt": legacy_mt,
                  "seedseqs": seedseqs}))
