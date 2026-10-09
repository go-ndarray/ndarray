# Generates testdata/values.json: `uv run --with numpy python testdata/values.py > testdata/values.json`
# Every number is a string (repr), so NaN, infinities and 64-bit integers
# survive JSON; complex numbers are [re, im].
import json, platform, warnings
import numpy as np

warnings.simplefilter("ignore")
DT = ['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64',
      'float32','float64','complex64','complex128']

def inputs(dt):
    d = np.dtype(dt)
    if d.kind == 'b':
        return [True, False, True, False, True, True], [True, True, False, False, False, True]
    if d.kind in 'iu':
        info = np.iinfo(d)
        x = [0, 1, 7, info.max, info.min, 3 if d.kind == 'u' else -3]
        y = [1, 3, 2, 2, 5, 4]
        return x, y
    if d.kind == 'f':
        return [0.0, -1.5, 2.25, float('nan'), float('inf'), -0.0], [1.0, 2.0, -0.5, 1.0, 3.0, 4.0]
    return [0j, 1.5-2j, -0.5+0.25j, complex('nan+1j'), 3+4j, -2-0j], [1+1j, 2+0j, -0.5j, 1+0j, 3-4j, 0.5+0.5j]

def enc(v):
    v = np.asarray(v)
    if v.dtype.kind == 'c':
        return [[repr(float(z.real)), repr(float(z.imag))] for z in v.ravel()]
    if v.dtype.kind == 'b':
        return [str(bool(z)).lower() for z in v.ravel()]
    if v.dtype.kind in 'iu':
        return [str(int(z)) for z in v.ravel()]
    return [repr(float(z)) for z in v.ravel()]

def rec(f):
    try:
        r = np.asarray(f())
        return {"dtype": str(r.dtype), "shape": list(r.shape), "data": enc(r)}
    except Exception as e:
        return {"error": type(e).__name__}

out = {"numpy": np.__version__, "machine": platform.machine(), "cases": {}}
for dt in DT:
    xs, ys = inputs(dt)
    x = np.array(xs, dtype=dt); y = np.array(ys, dtype=dt)
    c = {"x": enc(x), "y": enc(y)}
    m = np.array(xs, dtype=dt).reshape(2, 3)
    ops = {
        "add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y, "div": lambda: x / y,
        "maximum": lambda: np.maximum(x, y), "minimum": lambda: np.minimum(x, y),
        "eq": lambda: x == y, "ne": lambda: x != y, "lt": lambda: x < y, "le": lambda: x <= y,
        "gt": lambda: x > y, "ge": lambda: x >= y,
        "sqrt": lambda: np.sqrt(x), "exp": lambda: np.exp(x), "log": lambda: np.log(x),
        "sin": lambda: np.sin(x), "abs": lambda: np.abs(x), "neg": lambda: -x,
        "floor": lambda: np.floor(x), "square": lambda: np.square(x),
        "power": lambda: np.power(x, 2.5),
        "sum0": lambda: m.sum(axis=0), "sum1": lambda: m.sum(axis=1), "prod1": lambda: m.prod(axis=1),
        "max0": lambda: m.max(axis=0), "min1": lambda: m.min(axis=1),
        "argmax1": lambda: m.argmax(axis=1), "argmin0": lambda: m.argmin(axis=0),
        "cumsum1": lambda: np.cumsum(m, axis=1), "cumprod0": lambda: np.cumprod(m, axis=0),
        "mean1": lambda: m.mean(axis=1),
        "matmul": lambda: m @ m.T,
        "nonzero": lambda: np.flatnonzero(x),
        "where": lambda: np.where(x > y, x, y) if dt != 'bool' else np.where(x, x, y),
    }
    for to in DT:
        ops["astype_" + to] = (lambda to=to: x.astype(to))
    for k, f in ops.items():
        c[k] = rec(f)
    out["cases"][dt] = c
print(json.dumps(out, indent=0))
