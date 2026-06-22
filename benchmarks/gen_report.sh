#!/usr/bin/env bash
# Regenerate BENCHMARKS.md: verify correctness, run the Go benchmarks, run the
# NumPy benchmarks single-threaded (fair core) and multi-threaded (real-world),
# then merge into the parity table. All numbers are measured here, never edited
# by hand.
#
# Usage: benchmarks/gen_report.sh [path-to-python]
#   python defaults to /tmp/np-venv/bin/python (numpy 2.2.4 venv).
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${1:-/tmp/np-venv/bin/python}"
OUT="BENCHMARKS.md"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export CGO_ENABLED=0 GOWORK=off

echo ">> correctness verification (go vs numpy)"
GOND_VERIFY=1 go test -run TestDumpForVerify -count=1 ./benchmarks >/dev/null
VERIFY="$("$PY" benchmarks/verify_numpy.py)"
echo "   $VERIFY"
echo "$VERIFY" | grep -q "VERIFY OK" || { echo "correctness check FAILED"; exit 1; }

echo ">> go-ndarray benchmarks (multi-core, GOMAXPROCS=$(sysctl -n hw.ncpu))"
go test -run=XXX -bench=. -benchtime=2s -count=1 ./benchmarks | tee "$TMP/go.txt"

echo ">> numpy benchmarks (1-thread BLAS)"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$PY" benchmarks/bench_numpy.py | tee "$TMP/np1.txt"

echo ">> numpy benchmarks (default multi-thread BLAS)"
"$PY" benchmarks/bench_numpy.py | tee "$TMP/npN.txt"

echo ">> merging -> $OUT"
GO_VERSION="$(go version | awk '{print $3}')"
NP_VERSION="$(grep -m1 '^# numpy' "$TMP/np1.txt" | awk '{print $3}')"
NP_BLAS="$(grep -m1 '^# blas' "$TMP/np1.txt" | cut -d' ' -f3-)"
CPU="$(sysctl -n machdep.cpu.brand_string)"
NCPU="$(sysctl -n hw.ncpu)"

VERIFY="$VERIFY" GO_VERSION="$GO_VERSION" NP_VERSION="$NP_VERSION" \
NP_BLAS="$NP_BLAS" CPU="$CPU" NCPU="$NCPU" \
"$PY" benchmarks/merge_report.py "$TMP/go.txt" "$TMP/np1.txt" "$TMP/npN.txt" > "$OUT"
echo ">> done: $OUT"
