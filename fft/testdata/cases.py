# Generates fft/testdata/cases.json: `uv run --with numpy python fft/testdata/cases.py > fft/testdata/cases.json`
# Inputs are a fixed formula, not a random generator, and are stored with the
# results; numbers are repr strings, complex numbers [re, im].
import json, math
import numpy as np

def signal(shape, dt):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    re = np.sin(0.7 * i + 0.3) + 0.25 * np.cos(2.3 * i)
    im = np.cos(1.1 * i - 0.2) * 0.5
    if np.dtype(dt).kind == 'c':
        return (re + 1j * im).astype(dt).reshape(shape)
    if np.dtype(dt).kind == 'i':
        return np.round(re * 10).astype(dt).reshape(shape)
    return re.astype(dt).reshape(shape)

def enc(v):
    v = np.asarray(v)
    if v.dtype.kind == 'c':
        return [[repr(float(z.real)), repr(float(z.imag))] for z in v.ravel()]
    return [repr(float(z)) for z in v.ravel()]

calls = [
    ("fft", {}), ("fft", {"n": 6}), ("fft", {"n": 3, "axis": 0}), ("fft", {"norm": "ortho"}),
    ("ifft", {}), ("ifft", {"norm": "forward"}),
    ("rfft", {}), ("rfft", {"n": 7}), ("rfft", {"axis": 0, "norm": "ortho"}),
    ("irfft", {}), ("irfft", {"n": 9}), ("irfft", {"axis": 0}),
    ("hfft", {}), ("hfft", {"n": 5, "norm": "ortho"}), ("ihfft", {}), ("ihfft", {"norm": "forward"}),
    ("fft2", {}), ("fft2", {"s": [3, 6]}), ("ifft2", {"norm": "ortho"}),
    ("rfft2", {}), ("irfft2", {}), ("irfft2", {"s": [4, 7]}),
    ("fftn", {}), ("fftn", {"axes": [0]}), ("fftn", {"s": [2, 8], "axes": [0, -1]}),
    ("ifftn", {"axes": [-1, 0]}),
    ("rfftn", {}), ("rfftn", {"axes": [-1, 0]}), ("irfftn", {}), ("irfftn", {"s": [3, 5], "axes": [0, 1]}),
    ("fft", {"axis": 3}), ("fftn", {"axes": [0, 0]}), ("fft", {"n": 0}),
]
cases = []
for dt in ["float64", "float32", "complex128", "complex64", "int32"]:
    for shape in [(8,), (5,), (3, 4), (2, 3, 5)]:
        x = signal(shape, dt)
        for fn, kw in calls:
            if fn.endswith("2") and len(shape) < 2:
                continue
            c = {"fn": fn, "dtype": dt, "shape": list(shape), "x": enc(x),
                 "n": kw.get("n"), "s": kw.get("s"), "axis": kw.get("axis"),
                 "axes": kw.get("axes"), "norm": kw.get("norm")}
            try:
                r = np.asarray(getattr(np.fft, fn)(x, **kw))
                c.update(rdtype=str(r.dtype), rshape=list(r.shape), r=enc(r))
            except Exception as e:
                c["error"] = type(e).__name__
            cases.append(c)
for fn, kw in [("fftshift", {}), ("fftshift", {"axes": [1]}), ("ifftshift", {}), ("ifftshift", {"axes": [0]})]:
    for shape in [(5,), (3, 4), (2, 3, 5)]:
        x = signal(shape, "float64")
        if kw.get("axes") and kw["axes"][0] >= len(shape):
            continue
        r = getattr(np.fft, fn)(x, **kw)
        cases.append({"fn": fn, "dtype": "float64", "shape": list(shape), "x": enc(x), "axes": kw.get("axes"),
                      "rdtype": str(r.dtype), "rshape": list(r.shape), "r": enc(r)})
for n in [1, 4, 7]:
    for fn in ["fftfreq", "rfftfreq"]:
        r = getattr(np.fft, fn)(n, 0.1)
        cases.append({"fn": fn, "n": n, "rdtype": str(r.dtype), "rshape": list(r.shape), "r": enc(r)})
print(json.dumps({"numpy": np.__version__, "cases": cases}))
