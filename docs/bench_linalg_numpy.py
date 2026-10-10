# numpy side of docs/linalg.md: uv run --with numpy==2.5.3 python docs/bench_linalg_numpy.py
import time, sys, numpy as np, numpy.linalg as la
def mat(n, kind):
    r = np.random.default_rng(n)
    a = r.standard_normal((n, n))
    if kind == "spd": return a @ a.T + n * np.eye(n)
    if kind == "sym": return a + a.T
    return a
ops = [("solve", "", lambda a: la.solve(a, np.ones(a.shape[0]))), ("inv", "", la.inv),
       ("cholesky", "spd", la.cholesky), ("qr", "", la.qr), ("svd", "", lambda a: la.svd(a, full_matrices=False)),
       ("eigh", "sym", la.eigh), ("eig", "", la.eig)]
sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 256, 1024]
for name, kind, f in ops:
    for n in sizes:
        a = mat(n, kind)
        f(a)
        reps = max(1, int(0.3 / max(1e-6, (lambda t0: (f(a), time.perf_counter() - t0)[1])(time.perf_counter()))))
        reps = min(reps, 200)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); f(a); ts.append(time.perf_counter() - t0)
        print(f"{name}/{n} {np.median(ts)*1e3:.3f} ms")

# Stacks of small matrices.
rs=np.random.default_rng(9)
for cnt,n in [(10000,4),(1000,32)]:
    a=rs.standard_normal((cnt,n,n)); s=a+np.swapaxes(a,-1,-2)
    for name,f in [("inv",lambda: la.inv(a)),("det",lambda: la.det(a)),("eigh",lambda: la.eigh(s)),("svd",lambda: la.svd(a,full_matrices=False)),("eig",lambda: la.eig(a))]:
        f(); ts=[]
        for _ in range(7):
            t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
        print(f"{name}/{cnt}x{n} {np.median(ts)*1e3:.3f} ms")
