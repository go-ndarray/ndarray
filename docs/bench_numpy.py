#!/usr/bin/env python3
"""Benchmark numpy operations matched to the Go benchmarks in bench_test.go.

Reports ns/op (median over repeats) so the numbers are directly comparable to
`go test -bench` ns/op. Run on the same machine (the debian Tart VM, arm64) as
the Go benchmarks. numpy is single-threaded for these elementwise/reduction
ufuncs; matmul uses whatever BLAS numpy links (reported below).
"""
import timeit
import numpy as np


def bench(stmt, setup, number, repeat=7):
    t = timeit.Timer(stmt, setup=setup, globals=globals())
    # autorange would pick number; we fix it for stable per-op timing
    best = min(t.repeat(repeat=repeat, number=number)) / number
    return best * 1e9  # ns/op


def main():
    print(f"numpy {np.__version__}")
    try:
        cfg = np.show_config(mode="dicts")
        bl = cfg.get("Build Dependencies", {}).get("blas", {})
        print(f"blas: {bl.get('name', '?')}")
    except Exception:
        pass
    print()

    elem_sizes = [1 << 10, 1 << 14, 1 << 18, 1 << 22]
    print(f"{'op':<14}{'n':>10}{'ns/op':>16}")
    for n in elem_sizes:
        globals()['x'] = np.arange(n, dtype=np.float64) % 97 * 0.5
        globals()['y'] = np.arange(n, dtype=np.float64) % 97 * 0.5
        globals()['z'] = np.empty(n, dtype=np.float64)
        num = max(20, 2_000_000 // n)
        # Allocating ops (x + y returns a fresh array) AND the out= variants
        # (np.add(x, y, out=z)), which reuse z and so pay no allocation — the
        # apples-to-apples comparison for go-ndarray's *Into parity path. NumPy's
        # allocating form is already cheap because of its temp-array free-list.
        for name, stmt in [("Add", "x + y"), ("Mul", "x * y"),
                           ("Sqrt", "np.sqrt(x)"), ("Sum", "x.sum()"),
                           ("Max", "x.max()"),
                           ("AddInto", "np.add(x, y, out=z)"),
                           ("MulInto", "np.multiply(x, y, out=z)"),
                           ("SqrtInto", "np.sqrt(x, out=z)")]:
            print(f"{name:<14}{n:>10}{bench(stmt, '', num):>16.1f}")

    for n in [64, 128, 256, 512, 1024]:
        globals()['A'] = (np.arange(n * n, dtype=np.float64) % 101 * 0.25).reshape(n, n)
        globals()['B'] = (np.arange(n * n, dtype=np.float64) % 101 * 0.25).reshape(n, n)
        num = max(5, 50_000_000 // (n * n * n // 100))
        print(f"{'MatMul':<14}{n:>10}{bench('A @ B', '', num):>16.1f}")


if __name__ == "__main__":
    main()
