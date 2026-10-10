# Generates linalg/testdata/cases.json:
#   uv run --with numpy python linalg/testdata/cases.py > linalg/testdata/cases.json
# Inputs come from a seeded generator and are stored with numpy.linalg's
# results; numbers are repr strings, complex numbers [re, im]. "kappa" is the
# 2-norm condition number of the (first) matrix operand, which the Go test
# uses to scale its tolerance on values that depend on it.
import json, math
import numpy as np
import numpy.linalg as la

rng = np.random.default_rng(20261009)


def rand(*shape, dt="float64"):
    if np.dtype(dt).kind == "c":
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if np.dtype(dt).kind in "iu":
        return rng.integers(-5, 6, size=shape).astype(dt)
    if np.dtype(dt).kind == "b":
        return rng.integers(0, 2, size=shape).astype(bool)
    return rng.standard_normal(shape).astype(dt)


def spd(n, dt="float64", batch=()):
    a = rand(*batch, n, n, dt="complex128" if np.dtype(dt).kind == "c" else "float64")
    s = a @ np.conj(np.swapaxes(a, -1, -2)) + n * np.eye(n)
    return s.astype(dt)


def herm(n, dt="float64", batch=()):
    a = rand(*batch, n, n, dt=dt)
    return ((a + np.conj(np.swapaxes(a, -1, -2))) / 2).astype(dt)


def hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


def rankdef(m, n, r, dt="float64"):
    return (rand(m, r, dt=dt) @ rand(r, n, dt=dt)).astype(dt)


def singular(n, dt="float64"):
    a = rand(n, n, dt=dt)
    a[-1] = a[0]
    return a


def jordan(n, lam=2.0):
    return lam * np.eye(n) + np.eye(n, k=1)


def graded(n):
    d = 10.0 ** np.arange(-n // 2, n - n // 2)
    return d[:, None] * rand(n, n) / d[None, :]


def enc(v):
    v = np.asarray(v)
    if v.dtype.kind == "c":
        return [[repr(float(z.real)), repr(float(z.imag))] for z in v.ravel()]
    if v.dtype.kind == "b":
        return [repr(float(z)) for z in v.ravel()]
    return [repr(float(z)) for z in v.ravel()]


def arr(v):
    v = np.asarray(v)
    return {"dtype": str(v.dtype), "shape": list(v.shape), "data": enc(v)}


def kappa(a):
    a = np.asarray(a)
    if a.ndim < 2 or a.size == 0 or a.dtype.kind == "b":
        return 1.0
    try:
        with np.errstate(all="ignore"):
            c = np.max(la.cond(a.astype(np.complex128 if a.dtype.kind == "c" else np.float64)))
    except la.LinAlgError:
        return 1.0
    return float(c) if np.isfinite(c) else 1e300


def kind(e):
    m = str(e)
    if isinstance(e, la.LinAlgError):
        if "Singular" in m:
            return "singular"
        if "positive definite" in m:
            return "posdef"
        if "converge" in m:
            return "converge"
        if "infs or NaNs" in m:
            return "nonfinite"
        return "shape"
    if "norm order" in m or "Duplicate axes" in m or "Improper number" in m or type(e).__name__ == "AxisError" or "Expecting at least two" in m:
        return "args"
    return "shape"


cases = []
SHAPE_ONLY = {"qr": [0, 1], "svd": [0, 2], "eigh": [1], "eig": [1]}


prng = np.random.default_rng(1)  # perturbations, apart from the inputs' stream


def matching_move(w, w2):
    """Largest distance from an eigenvalue of w to its nearest unused one of w2."""
    w2 = list(w2)
    move = 0.0
    for z in w:
        j = min(range(len(w2)), key=lambda k: abs(w2[k] - z))
        move = max(move, abs(w2[j] - z))
        w2.pop(j)
    return move


def eig_tol(a):
    """100 times how far the eigenvalues move when each entry is perturbed
    by 1e-14 of itself (or of the largest entry, for a zero entry): an
    empirical bound on their sensitivity that respects the scaling
    balancing exploits."""
    a = np.asarray(a)
    if a.ndim < 2 or a.size == 0 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        return 0.0
    a = a.astype(np.complex128 if a.dtype.kind == "c" else np.float64)
    move = 0.0
    for m in a.reshape(-1, a.shape[-2], a.shape[-1]):
        size = np.where(m != 0, np.abs(m), np.abs(m).max())
        pert = m + 1e-14 * size * prng.standard_normal(m.shape)
        try:
            move = max(move, matching_move(la.eigvals(m), la.eigvals(pert)))
        except la.LinAlgError:
            return float("inf")
    return 100 * move


def effective_kappa(a, rcond):
    """sigma_max / sigma_r over the singular values above the cutoff."""
    a = np.asarray(a)
    if a.size == 0 or a.ndim < 2:
        return 1.0
    s = la.svd(a.astype(np.complex128 if a.dtype.kind == "c" else np.float64), compute_uv=False)
    s = s.reshape(-1, s.shape[-1])
    k = 1.0
    for row in s:
        keep = row[row > rcond * row[0]]
        if len(keep):
            k = max(k, keep[0] / keep[-1])
    return float(k)


def add(fn, inputs, args=None, tag="", **check):
    args = args or {}
    c = {"fn": fn, "tag": tag, "args": args, "inputs": [arr(x) for x in inputs], "kappa": kappa(inputs[0])}
    if fn in ("eig", "eigvals"):
        c["etol"] = eig_tol(inputs[0])
    if fn == "lstsq":
        a = np.asarray(inputs[0])
        if a.ndim == 2:
            rc = args.get("rcond")
            if rc is None:
                rc = np.finfo(np.float64).eps * max(a.shape)
            elif rc <= 0 or rc >= 1:
                rc = np.finfo(np.float64).eps / 2
            c["kappa"] = effective_kappa(a, rc)
    if fn == "pinv":
        c["kappa"] = effective_kappa(inputs[0], args.get("rcond", 1e-15))
    c.update(check)
    try:
        out = call(fn, inputs, args)
        c["outputs"] = [arr(o) for o in out]
        # Factors that are not unique are checked through invariants; only
        # their dtype and shape are kept.
        for i in SHAPE_ONLY.get(fn, []) if args.get("mode") not in ("raw", "r") else []:
            c["outputs"][i]["data"] = None
    except Exception as e:  # numpy refused: the Go side must refuse too
        c["error"] = kind(e)
        c["message"] = f"{type(e).__name__}: {e}"
    cases.append(c)


def ordarg(o):
    if o is None:
        return None
    if isinstance(o, str):
        return o
    return float(o)


def call(fn, x, a):
    if fn == "solve":
        return [la.solve(x[0], x[1])]
    if fn == "inv":
        return [la.inv(x[0])]
    if fn == "det":
        return [la.det(x[0])]
    if fn == "slogdet":
        return list(la.slogdet(x[0]))
    if fn == "cholesky":
        return [la.cholesky(x[0], upper=a.get("upper", False))]
    if fn == "qr":
        r = la.qr(x[0], mode=a["mode"])
        return [r] if a["mode"] == "r" else list(r)
    if fn == "svd":
        return list(la.svd(x[0], full_matrices=a["full"]))
    if fn == "svdvals":
        return [la.svdvals(x[0])]
    if fn == "eigh":
        return list(la.eigh(x[0], UPLO=a["uplo"]))
    if fn == "eigvalsh":
        return [la.eigvalsh(x[0], UPLO=a["uplo"])]
    if fn == "eig":
        return list(la.eig(x[0]))
    if fn == "eigvals":
        return [la.eigvals(x[0])]
    if fn == "lstsq":
        rc = a.get("rcond")
        xs, res, rank, s = la.lstsq(x[0], x[1], rcond=rc)
        return [xs, res, np.int64(rank), s]
    if fn == "pinv":
        if "rcond" in a:
            return [la.pinv(x[0], rcond=a["rcond"])]
        return [la.pinv(x[0])]
    if fn == "matrix_rank":
        return [np.asarray(la.matrix_rank(x[0], tol=a.get("tol")), dtype=np.int64)]
    if fn == "norm":
        ax = a.get("axis")
        if isinstance(ax, list):
            ax = tuple(ax)
        o = a.get("ord")
        if o in ("inf", "-inf"):
            o = float(o)
        return [np.asarray(la.norm(x[0], ord=o, axis=ax, keepdims=a.get("keepdims", False)))]
    if fn == "cond":
        o = a.get("p")
        if o in ("inf", "-inf"):
            o = float(o)
        return [np.asarray(la.cond(x[0], o))]
    if fn == "matrix_power":
        return [la.matrix_power(x[0], a["n"])]
    if fn == "multi_dot":
        return [np.asarray(la.multi_dot(x))]
    raise ValueError(fn)


F, F32, C, C64, I = "float64", "float32", "complex128", "complex64", "int32"

# --- solve
for dt in [F, F32, C, C64, I]:
    for n in [1, 3, 8]:
        a = rand(n, n, dt=dt)
        if dt == I:
            a = a + 10 * np.eye(n, dtype=a.dtype)
        add("solve", [a, rand(n, dt=dt)], tag=f"vec-{dt}")
        add("solve", [a, rand(n, 2, dt=dt)], tag=f"mat-{dt}")
add("solve", [rand(2, 3, 4, 4), rand(3, 4, 2)], tag="broadcast")
add("solve", [rand(3, 4, 4), rand(4)], tag="batched-vec")
add("solve", [rand(2, 1, 4, 4), rand(5, 4, 1)], tag="broadcast-ones")
add("solve", [rand(4, 4), rand(4, 0)], tag="no-rhs")
add("solve", [np.zeros((0, 0)), np.zeros(0)], tag="empty")
add("solve", [hilbert(8), np.ones(8)], tag="hilbert")
add("solve", [rand(4, 4, dt=F), rand(4, 2, dt=C)], tag="mixed")
add("solve", [rand(4, 4, dt=F32), rand(4, dt=I)], tag="f32-int")
add("solve", [singular(5), rand(5)], tag="singular")
add("solve", [np.zeros((3, 3)), rand(3)], tag="zero")
add("solve", [rand(3, 4), rand(3)], tag="nonsquare")
add("solve", [rand(3, 3), rand(4)], tag="badrows")
add("solve", [rand(3, 3), rand(4, 2)], tag="badrows2")
add("solve", [rand(2, 3, 3), rand(3, 3, 1)], tag="nobroadcast")
add("solve", [rand(3), rand(3)], tag="1d")
add("solve", [rand(3, 3), np.float64(1.0)], tag="0d-b")
add("solve", [np.eye(3).copy(), np.array([1.0, np.nan, 2.0])], tag="nan")

# --- inv, det, slogdet
mats = [(rand(n, n, dt=dt), f"{dt}-{n}") for dt in [F, F32, C, C64] for n in [0, 1, 2, 5, 12]]
mats += [(rand(4, 4, dt=I), "int"), (rand(3, 3, dt="bool") | np.eye(3, dtype=bool), "bool"),
         (rand(2, 3, 4, 4), "batched"), (rand(0, 3, 3), "empty-batch"), (hilbert(7), "hilbert"),
         (graded(6), "graded"), (rand(40, 40), "n40"), (rand(34, 34, dt=C), "c34")]
for a, tag in mats:
    add("inv", [a], tag=tag)
    add("det", [a], tag=tag)
    add("slogdet", [a], tag=tag)
for a, tag in [(singular(5), "singular"), (np.zeros((3, 3)), "zero"), (np.zeros((2, 2, 2)), "zero-batch"),
               (singular(4, dt=C), "singular-c"), (np.array([[1.0, 2.0], [2.0, 4.0]]), "rank1")]:
    add("inv", [a], tag=tag)
    add("det", [a], tag=tag)
    add("slogdet", [a], tag=tag)
scaled = np.diag([1e200, 1e200, 1e-300])
add("det", [scaled], tag="scaled")
add("slogdet", [scaled], tag="scaled")
add("det", [np.diag([1e200, 1e200, 1e200])], tag="overflow")
add("det", [np.diag([1e-200, 1e-200, 1e-200])], tag="underflow")
add("slogdet", [np.diag([1e-200, 1e-200, 1e-200])], tag="underflow")
for f in ["inv", "det", "slogdet"]:
    add(f, [rand(3, 4)], tag="nonsquare")
    add(f, [rand(3)], tag="1d")
add("inv", [np.array([[np.nan, 0.0], [0.0, 1.0]])], tag="nan")

# --- cholesky
for dt in [F, F32, C, C64]:
    for n in [0, 1, 2, 5, 12] + ([34] if dt in (F, C) else []):
        a = spd(n, dt)
        add("cholesky", [a], tag=f"{dt}-{n}")
        add("cholesky", [a], {"upper": True}, tag=f"{dt}-{n}-up")
g = spd(6)
gl, gu = g.copy(), g.copy()
gl[np.triu_indices(6, 1)] = 99.0  # garbage numpy must not read
gu[np.tril_indices(6, -1)] = -99.0
add("cholesky", [gl], tag="lower-only")
add("cholesky", [gu], {"upper": True}, tag="upper-only")
add("cholesky", [spd(4, batch=(2, 3))], tag="batched")
add("cholesky", [spd(3, C, batch=(2,))], {"upper": True}, tag="batched-c-up")
add("cholesky", [np.array([[4, 2], [2, 3]], dtype=np.int64)], tag="int")
add("cholesky", [hilbert(8)], tag="hilbert")
add("cholesky", [np.array([[1.0, 2.0], [2.0, 1.0]])], tag="indefinite")
add("cholesky", [np.array([[0.0, 0.0], [0.0, 1.0]])], tag="semidefinite")
add("cholesky", [np.array([[1.0, 0.0], [0.0, -1.0]])], tag="negative")
add("cholesky", [np.array([[np.nan, 0.0], [0.0, 1.0]])], tag="nan")
add("cholesky", [rand(3, 4)], tag="nonsquare")

# --- qr
shapes = [(5, 3), (3, 5), (4, 4), (1, 1), (0, 3), (3, 0), (2, 4, 3)]
big = {F: [(36, 20), (20, 36)], C: [(36, 20), (20, 36)]}
for mode in ["reduced", "complete", "r", "raw"]:
    for dt in [F, C, F32, C64]:
        for s in shapes + big.get(dt, []):
            add("qr", [rand(*s, dt=dt)], {"mode": mode}, tag=f"{dt}-{s}")
    add("qr", [rankdef(6, 5, 2)], {"mode": mode}, tag="rankdef")
    add("qr", [np.zeros((3, 2))], {"mode": mode}, tag="zero")
    add("qr", [rand(4, 3, dt=I)], {"mode": mode}, tag="int")
    add("qr", [hilbert(8)], {"mode": mode}, tag="hilbert")
add("qr", [rand(3)], {"mode": "reduced"}, tag="1d")

# --- svd, svdvals
for full in [True, False]:
    for dt in [F, C, F32, C64]:
        for s in shapes + big.get(dt, []):
            add("svd", [rand(*s, dt=dt)], {"full": full}, tag=f"{dt}-{s}")
    add("svd", [rankdef(7, 5, 2)], {"full": full}, tag="rankdef")
    add("svd", [rankdef(5, 7, 3, dt=C)], {"full": full}, tag="rankdef-c")
    add("svd", [np.zeros((4, 3))], {"full": full}, tag="zero")
    add("svd", [hilbert(10)], {"full": full}, tag="hilbert")
    add("svd", [graded(8)], {"full": full}, tag="graded")
    add("svd", [np.diag([3.0, 3.0, 1.0, 1.0])], {"full": full}, tag="repeated")
    add("svd", [rand(4, 3, dt=I)], {"full": full}, tag="int")
    add("svd", [rand(3)], {"full": full}, tag="1d")
add("svd", [np.array([[np.nan, 0.0], [0.0, 1.0]])], {"full": True}, tag="nan")
for s in shapes + [(9, 9)]:
    add("svdvals", [rand(*s)], tag=f"{s}")
add("svdvals", [rand(6, 4, dt=C64)], tag="c64")
add("svdvals", [hilbert(12)], tag="hilbert")

# --- eigh, eigvalsh
for uplo in ["L", "U"]:
    for dt in [F, C, F32, C64]:
        for n in [0, 1, 2, 5, 12] + ([34] if dt in (F, C) else []):
            a = herm(n, dt)
            add("eigh", [a], {"uplo": uplo}, tag=f"{dt}-{n}")
            add("eigvalsh", [a], {"uplo": uplo}, tag=f"{dt}-{n}")
    g = herm(6, C)
    if uplo == "L":
        g[np.triu_indices(6, 1)] = 7 + 7j
    else:
        g[np.tril_indices(6, -1)] = -7 - 7j
    add("eigh", [g], {"uplo": uplo}, tag="one-triangle")
    add("eigh", [np.eye(5)], {"uplo": uplo}, tag="identity")
    add("eigh", [np.diag([2.0, 1.0, 2.0, 1.0, 2.0])], {"uplo": uplo}, tag="repeated")
    add("eigh", [herm(4, batch=(2, 3))], {"uplo": uplo}, tag="batched")
    add("eigh", [hilbert(10)], {"uplo": uplo}, tag="hilbert")
    add("eigh", [np.array([[2, 1], [1, 2]], dtype=np.int64)], {"uplo": uplo}, tag="int")
    add("eigh", [np.diag([1e300, 1e-300, 1.0])], {"uplo": uplo}, tag="wide-range")
    add("eigh", [1e-310 * herm(4)], {"uplo": uplo}, tag="tiny")
    add("eigh", [1e305 * herm(4)], {"uplo": uplo}, tag="huge")
    add("eigh", [np.diag(np.ones(7)) + np.diag(1e-9 * np.ones(6), 1) + np.diag(1e-9 * np.ones(6), -1)], {"uplo": uplo}, tag="cluster")
add("eigh", [rand(3, 4)], {"uplo": "L"}, tag="nonsquare")
add("eigvalsh", [rand(3)], {"uplo": "L"}, tag="1d")

# --- eig, eigvals
emats = [(rand(n, n, dt=dt), f"{dt}-{n}") for dt in [F, C, F32, C64] for n in [0, 1, 2, 5, 12] + ([34] if dt in (F, C) else [])]
emats += [(jordan(4), "jordan"), (np.array([[1.0, 1.0], [0.0, 1.0]]), "defective2"),
          (np.roll(np.eye(5), 1, axis=1), "permutation"), (np.triu(rand(6, 6)), "triangular"),
          (np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation"), (graded(8), "graded"),
          (rand(2, 3, 4, 4), "batched"), (rand(4, 4, dt=I), "int"), (np.zeros((3, 3)), "zero"),
          (herm(6), "symmetric"), (np.diag([1.0, 2.0, 3.0]) + np.diag([1e-8, 1e-8], 1), "nearly-diag"),
          (np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [4.0, 5.0, 6.0]]), "isolated"),
          (np.vander(np.arange(1.0, 7.0)), "vander"), (np.ones((5, 5)), "ones"),
          (np.array([[1e-300, 1.0], [-1.0, 1e300]]), "extreme")]
for a, tag in emats:
    add("eig", [a], tag=tag)
    add("eigvals", [a], tag=tag)
for f in ["eig", "eigvals"]:
    add(f, [np.array([[np.nan, 0.0], [0.0, 1.0]])], tag="nan")
    add(f, [np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=C)], tag="inf")
    add(f, [rand(3, 4)], tag="nonsquare")

# --- lstsq
for dt in [F, C, F32]:
    for m, n in [(6, 3), (3, 6), (4, 4), (1, 1)]:
        a = rand(m, n, dt=dt)
        add("lstsq", [a, rand(m, dt=dt)], tag=f"{dt}-{m}x{n}-vec")
        add("lstsq", [a, rand(m, 2, dt=dt)], tag=f"{dt}-{m}x{n}-mat")
for rc in [None, -1.0, 1e-3, 0.5, 2.0]:
    add("lstsq", [rankdef(6, 4, 2), rand(6, 2)], {"rcond": rc}, tag=f"rankdef-{rc}")
add("lstsq", [hilbert(8)[:, :6], np.ones(8)], tag="hilbert")
add("lstsq", [rand(5, 3), rand(5, 0)], tag="no-rhs")
add("lstsq", [np.zeros((0, 3)), np.zeros(0)], tag="m0")
add("lstsq", [np.zeros((4, 3)), rand(4)], tag="zero")
add("lstsq", [rand(5, 3, dt=I), rand(5, dt=I)], tag="int")
add("lstsq", [rand(2, 5, 3), rand(2, 5)], tag="3d")
add("lstsq", [rand(5, 3), rand(4)], tag="badrows")
add("lstsq", [rand(5, 3), rand(5, 2, 2)], tag="3d-b")

# --- pinv
for s in [(5, 3), (3, 5), (4, 4), (2, 3, 4), (1, 1)]:
    for dt in [F, C, F32]:
        add("pinv", [rand(*s, dt=dt)], tag=f"{dt}-{s}")
add("pinv", [rankdef(6, 5, 2)], tag="rankdef")
add("pinv", [rankdef(6, 5, 2)], {"rcond": 1e-3}, tag="rankdef-rcond")
add("pinv", [hilbert(8)], tag="hilbert")
add("pinv", [np.zeros((3, 2))], tag="zero")
add("pinv", [np.zeros((0, 3), dtype=np.int64)], tag="empty-int")
add("pinv", [rand(4, dt=F)], tag="1d")

# --- matrix_rank
for a, tag in [(rand(5, 4), "full"), (rankdef(6, 5, 2), "rankdef"), (np.zeros((3, 3)), "zero"),
               (hilbert(12), "hilbert"), (rankdef(5, 5, 3, dt=C), "complex"), (rand(2, 3, 4), "batched"),
               (rand(5, 5, dt=F32), "f32"), (rankdef(6, 6, 3).astype(F32), "f32-rankdef"),
               (np.ones(4), "1d"), (np.zeros(3), "1d-zero"), (np.float64(0.5), "0d"), (np.eye(0), "empty")]:
    add("matrix_rank", [a], tag=tag)
add("matrix_rank", [hilbert(12)], {"tol": 1e-8}, tag="tol")

# --- norm
vec = rand(7)
vecs = [(vec, "f64"), (rand(6, dt=C), "c128"), (rand(5, dt=F32), "f32"), (rand(6, dt=I), "int"),
        (np.zeros(0), "empty"), (np.array([0.0, 3.0, -4.0, 0.0]), "zeros")]
vords = [None, "inf", "-inf", 0, 1, -1, 2, -2, 3, 0.5, "fro", "nuc"]
for v, tag in vecs:
    for o in vords:
        add("norm", [v], {"ord": ordarg(o)}, tag=f"vec-{tag}")
mat = rand(4, 5)
mords = [None, "fro", "nuc", "inf", "-inf", 1, -1, 2, -2, 3, 0]
for m_, tag in [(mat, "f64"), (rand(3, 3, dt=C), "c128"), (rand(4, 2, dt=C64), "c64"), (rand(3, 4, dt=I), "int"),
                (np.zeros((0, 3)), "empty-rows"), (np.zeros((3, 0)), "empty-cols")]:
    for o in mords:
        add("norm", [m_], {"ord": ordarg(o)}, tag=f"mat-{tag}")
t3 = rand(2, 3, 4)
add("norm", [t3], {}, tag="3d-default")
add("norm", [t3], {"ord": 1.0}, tag="3d-ord")
for ax in [0, 1, -1, 2, [0, 2], [2, 0], [-1, -2], [1, 1], [0, 1, 2], 3, -4]:
    for o in [None, 1, "inf", 2, "fro", "nuc"]:
        add("norm", [t3], {"ord": ordarg(o), "axis": ax}, tag="3d-axis")
for ax in [None, 1, [0, 2]]:
    add("norm", [t3], {"axis": ax, "keepdims": True}, tag="keepdims")
    add("norm", [t3], {"axis": ax, "keepdims": True, "ord": "inf" if ax == 1 else None}, tag="keepdims-ord")
add("norm", [mat], {"ord": "fro", "keepdims": True}, tag="keepdims-2d")
add("norm", [np.zeros((0, 3, 0))], {"ord": -1.0, "axis": [1, 2]}, tag="empty-batch-bad")
add("norm", [np.zeros((0, 3, 3))], {"ord": -1.0, "axis": [1, 2]}, tag="empty-batch")
add("norm", [np.array([1e200, 1e200])], {}, tag="overflow")
add("norm", [np.float64(-3.0)], {}, tag="0d")

# --- cond
for p in [None, 2, -2, 1, -1, "inf", "-inf", "fro", "nuc", 3]:
    add("cond", [rand(4, 4)], {"p": ordarg(p)}, tag="f64")
    add("cond", [rand(3, 3, dt=C)], {"p": ordarg(p)}, tag="c128")
    add("cond", [hilbert(6)], {"p": ordarg(p)}, tag="hilbert")
    add("cond", [singular(4)], {"p": ordarg(p)}, tag="singular")
    add("cond", [rand(2, 3, 3, dt=F32)], {"p": ordarg(p)}, tag="batched-f32")
    add("cond", [rand(4, 3)], {"p": ordarg(p)}, tag="nonsquare")
add("cond", [np.zeros((3, 3))], {}, tag="zero")
add("cond", [np.zeros((0, 0))], {}, tag="empty")
add("cond", [np.array([[np.nan, 0.0], [0.0, 1.0]])], {"p": 1.0}, tag="nan")
add("cond", [rand(3)], {}, tag="1d")

# --- matrix_power
for n in [0, 1, 2, 3, 5, 8, -1, -3]:
    add("matrix_power", [rand(4, 4)], {"n": n}, tag="f64")
    add("matrix_power", [rand(3, 3, dt=C)], {"n": n}, tag="c128")
    add("matrix_power", [rand(2, 3, 3)], {"n": n}, tag="batched")
for n in [0, 1, 2, 3, 6]:
    add("matrix_power", [rand(3, 3, dt="int64")], {"n": n}, tag="int")
    add("matrix_power", [rand(3, 3, dt=F32)], {"n": n}, tag="f32")
add("matrix_power", [singular(3)], {"n": -1}, tag="singular")
add("matrix_power", [rand(0, 3, 3)], {"n": 4}, tag="empty-batch")
add("matrix_power", [rand(3, 4)], {"n": 2}, tag="nonsquare")

# --- multi_dot
add("multi_dot", [rand(3, 4), rand(4, 5)], tag="two")
add("multi_dot", [rand(10, 2), rand(2, 30), rand(30, 4)], tag="three")
add("multi_dot", [rand(2, 30), rand(30, 4), rand(4, 25)], tag="three-b")
add("multi_dot", [rand(5, 3), rand(3, 40), rand(40, 2), rand(2, 6)], tag="four")
add("multi_dot", [rand(3), rand(3, 4), rand(4, 5), rand(5)], tag="vec-vec")
add("multi_dot", [rand(3), rand(3, 4), rand(4, 5)], tag="vec-first")
add("multi_dot", [rand(3, 4), rand(4, 5), rand(5, 6), rand(6)], tag="vec-last")
add("multi_dot", [rand(3, 4, dt=C), rand(4, 5), rand(5, 2, dt=I), rand(2, 2)], tag="mixed")
add("multi_dot", [rand(3, 4)], tag="one")
add("multi_dot", [rand(3, 4), rand(5, 6), rand(6, 2)], tag="mismatch")
add("multi_dot", [rand(3, 4), rand(2, 4, 5), rand(5, 2)], tag="3d")
add("multi_dot", [rand(3, 4), rand(5, 2)], tag="two-mismatch")

print(json.dumps({"numpy": np.__version__, "cases": cases}))
