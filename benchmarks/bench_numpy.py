#!/usr/bin/env python3
"""NumPy side of the go-ndarray performance-parity harness.

Mirrors benchmarks/bench_test.go op-for-op, shape-for-shape, dtype float64,
with the identical deterministic data fill, so the two are directly comparable.

Emits tab-separated rows `name<TAB>ns_per_op` on stdout (one per (op, shape)),
preceded by lines starting with `#` (metadata) which the report generator
ignores. ns/op is min-of-`repeat` over `number` inner iterations — the standard
robust timer, comparable to Go's reported ns/op.

Thread control is the caller's job (env vars). Run it twice: once pinned to a
single BLAS thread (the fair core comparison vs go-ndarray's own kernels) and
once at NumPy's default thread count (the real-world Accelerate/OpenBLAS row).
"""
import os
import sys
import timeit

import numpy as np


def bench(stmt, number, repeat=7):
    t = timeit.Timer(stmt, globals=globals())
    best = min(t.repeat(repeat=repeat, number=number)) / number
    return best * 1e9  # ns/op


def emit(name, ns):
    print(f"{name}\t{ns:.1f}")
    sys.stdout.flush()


def main():
    print(f"# numpy {np.__version__}")
    try:
        cfg = np.show_config(mode="dicts")
        bl = cfg.get("Build Dependencies", {}).get("blas", {})
        print(f"# blas {bl.get('name', '?')} {bl.get('version', '')}")
    except Exception:
        pass
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        print(f"# env {k}={os.environ.get(k, '<unset>')}")

    elem = [1 << 10, 1 << 14, 1 << 18, 1 << 22]
    g = globals()

    for n in elem:
        g["x"] = np.arange(n, dtype=np.float64) % 97 * 0.5
        g["y"] = np.arange(n, dtype=np.float64) % 97 * 0.5
        g["z"] = np.empty(n, dtype=np.float64)
        num = max(20, 2_000_000 // n)
        for name, stmt in [
            ("Add", "x + y"),
            ("Mul", "x * y"),
            ("Exp", "np.exp(x)"),
            ("Sqrt", "np.sqrt(x)"),
            ("AddInto", "np.add(x, y, out=z)"),
            ("MulInto", "np.multiply(x, y, out=z)"),
            ("SqrtInto", "np.sqrt(x, out=z)"),
            ("Sum", "x.sum()"),
            ("Mean", "x.mean()"),
            ("Max", "x.max()"),
        ]:
            emit(f"{name}/n={n}", bench(stmt, num))

    # axis reductions / broadcasting / views / concat over a 1024x1024 matrix
    g["M"] = (np.arange(1024 * 1024, dtype=np.float64) % 101 * 0.25).reshape(1024, 1024)
    g["row"] = (np.arange(1024, dtype=np.float64) % 101 * 0.25).reshape(1, 1024)
    emit("SumAxis0", bench("M.sum(axis=0)", 200))
    emit("SumAxis1", bench("M.sum(axis=1)", 200))
    emit("MaxAxis1", bench("M.max(axis=1)", 200))
    emit("BroadcastAdd", bench("M + row", 200))
    emit("SliceView", bench("M[0:1024:2, 100:900]", 200000))
    g["V"] = M[0:1024:2, 100:900]
    emit("SliceMaterialize", bench("np.ascontiguousarray(V)", 2000))

    g["A2"] = (np.arange(512 * 1024, dtype=np.float64) % 101 * 0.25).reshape(512, 1024)
    g["B2"] = (np.arange(512 * 1024, dtype=np.float64) % 101 * 0.25).reshape(512, 1024)
    emit("ConcatAxis0", bench("np.concatenate([A2, B2], axis=0)", 2000))
    emit("Stack", bench("np.stack([A2, B2], axis=0)", 2000))

    # dot / inner / outer
    g["u"] = np.arange(1 << 20, dtype=np.float64) % 97 * 0.5
    g["w"] = np.arange(1 << 20, dtype=np.float64) % 97 * 0.5
    emit("Dot1D", bench("np.dot(u, w)", 2000))

    g["Mv"] = (np.arange(1024 * 1024, dtype=np.float64) % 101 * 0.25).reshape(1024, 1024)
    g["vv"] = np.arange(1024, dtype=np.float64) % 97 * 0.5
    emit("MatVec", bench("Mv @ vv", 5000))

    g["I1"] = (np.arange(512 * 512, dtype=np.float64) % 101 * 0.25).reshape(512, 512)
    g["I2"] = (np.arange(512 * 512, dtype=np.float64) % 101 * 0.25).reshape(512, 512)
    emit("Inner", bench("np.inner(I1, I2)", 30))

    g["o1"] = np.arange(2048, dtype=np.float64) % 97 * 0.5
    g["o2"] = np.arange(2048, dtype=np.float64) % 97 * 0.5
    emit("Outer", bench("np.outer(o1, o2)", 2000))

    # matmul (square + non-square)
    for n in [128, 256, 512, 1024]:
        g["P"] = (np.arange(n * n, dtype=np.float64) % 101 * 0.25).reshape(n, n)
        g["Q"] = (np.arange(n * n, dtype=np.float64) % 101 * 0.25).reshape(n, n)
        num = max(5, 50_000_000 // (n * n * n // 100))
        emit(f"MatMul/n={n}", bench("P @ Q", num))

    g["R1"] = (np.arange(1024 * 256, dtype=np.float64) % 101 * 0.25).reshape(1024, 256)
    g["R2"] = (np.arange(256 * 1024, dtype=np.float64) % 101 * 0.25).reshape(256, 1024)
    emit("MatMulNonSquare", bench("R1 @ R2", 50))


if __name__ == "__main__":
    main()
