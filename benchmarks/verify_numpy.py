#!/usr/bin/env python3
"""Confirm go-ndarray agrees with NumPy on /tmp/gond_verify.txt before timing.

Recomputes the same ops with the identical deterministic data and checks each
scalar / array-checksum against the Go dump within a tight relative tolerance.
Exits non-zero on any mismatch, so gen_report.sh aborts rather than publishing
a parity table built on wrong numbers.

Operands are rebuilt fresh for every op (no shared, possibly-aliased buffers),
and spurious Accelerate fast-path FP warnings on macOS are silenced.
"""
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
np.seterr(all="ignore")


def vec(n):
    return np.arange(n, dtype=np.float64) % 97 * 0.5


def mat(r, c):
    return (np.arange(r * c, dtype=np.float64) % 101 * 0.25).reshape(r, c)


def main():
    got = {}
    with open("/tmp/gond_verify.txt") as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            got[k] = v

    exp = {}

    def arr(tag, a):
        a = np.ascontiguousarray(a).ravel()
        exp[tag + ".sum"] = float(a.sum())
        exp[tag + ".first"] = float(a[0])
        exp[tag + ".last"] = float(a[-1])
        exp[tag + ".mid"] = float(a[a.size // 2])

    n = 1 << 16
    arr("add", vec(n) + vec(n))
    arr("mul", vec(n) * vec(n))
    arr("sqrt", np.sqrt(vec(n)))
    arr("exp", np.exp(vec(n)))
    exp["sum"] = float(vec(n).sum())
    exp["mean"] = float(vec(n).mean())
    exp["max"] = float(vec(n).max())
    exp["dot"] = float(np.dot(vec(n), vec(n)))

    arr("matmul", mat(256, 256) @ mat(256, 256))
    arr("matmul_ns", mat(300, 128) @ mat(128, 220))
    arr("sumaxis0", mat(256, 256).sum(axis=0))
    arr("sumaxis1", mat(256, 256).sum(axis=1))
    arr("maxaxis1", mat(256, 256).max(axis=1))
    arr("bcast", mat(256, 256) + mat(1, 256))
    arr("outer", np.outer(vec(700), vec(900)))
    arr("inner", np.inner(mat(100, 64), mat(80, 64)))
    arr("slice", mat(256, 256)[0:256:2, 50:200])

    fails = 0
    for k, e in exp.items():
        if k not in got:
            print(f"MISSING {k}")
            fails += 1
            continue
        g = float(got[k])
        rel = abs(g - e) / max(abs(e), 1.0)
        if rel > 1e-12:
            print(f"MISMATCH {k}: go={g!r} numpy={e!r} rel={rel:.2e}")
            fails += 1

    if fails:
        print(f"\nVERIFY FAILED: {fails} mismatch(es)")
        sys.exit(1)
    print(f"VERIFY OK: {len(exp)} checks agree with NumPy (rel tol 1e-12)")


if __name__ == "__main__":
    main()
