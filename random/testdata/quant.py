# Writes NumPy's output for 10^6 draws of the samplers that call the C library,
# as little-endian float64 files, for TestQuantifyLibm (docs/random.md quotes it):
#   mkdir /tmp/q && uv run --python cpython-3.12-macos-x86_64 --with numpy==2.5.3 \
#       python random/testdata/quant.py /tmp/q
#   RANDOM_QUANT_DIR=/tmp/q go test -run TestQuantifyLibm -v ./random
import numpy as np, sys, json
from numpy.random import Generator, PCG64, RandomState
N=1_000_000
d=sys.argv[1]
jobs=[]
def save(name, v):
    np.ascontiguousarray(v, dtype='<f8').tofile(f"{d}/{name}.bin"); jobs.append(name)
save("legacy_randn", RandomState(0).randn(N))
save("legacy_normal", RandomState(0).normal(1.0, 3.0, N))
save("legacy_gamma3", RandomState(0).standard_gamma(3.0, N))
save("legacy_exp", RandomState(0).standard_exponential(N))
g=lambda: Generator(PCG64(0))
save("gamma05", g().standard_gamma(0.5, N))
save("gamma3", g().standard_gamma(3.0, N))
save("beta23", g().beta(2.0, 3.0, N))
save("beta05", g().beta(0.5, 0.5, N))
save("lognormal", g().lognormal(0.5, 0.25, N))
save("laplace", g().laplace(0.0, 1.0, N))
save("logistic", g().logistic(0.0, 1.0, N))
save("gumbel", g().gumbel(0.0, 1.0, N))
save("exp_inv", g().standard_exponential(N, method='inv'))
save("pareto", g().pareto(3.0, N))
save("weibull", g().weibull(2.0, N))
save("vonmises", g().vonmises(0.5, 2.0, N))
save("chisquare", g().chisquare(3.0, N))
save("standard_t", g().standard_t(4.0, N))
save("normal", g().standard_normal(N))
save("binomial", g().binomial(1000, 0.4, N).astype('<f8'))
save("poisson", g().poisson(50.0, N).astype('<f8'))
save("poisson3", g().poisson(3.0, N).astype('<f8'))
save("geometric", g().geometric(0.05, N).astype('<f8'))
json.dump(jobs, open(f"{d}/jobs.json","w"))
