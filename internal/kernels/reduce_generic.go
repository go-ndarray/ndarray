//go:build !arm64 && !amd64

package kernels

// On architectures without a generated SIMD reduction kernel, the dispatch
// wrapper routes to the scalar oracle. sumSIMD aliases it so the
// SIMD-vs-scalar test (which then trivially holds, scalar == scalar) still
// exercises this path and the package builds uniformly across all six targets.
//
// amd64 (SSE2) and arm64 (NEON) ship kernels (see reduce_amd64.go /
// reduce_arm64.go), so this file is the loong64/ppc64le/riscv64/s390x fallback.
// For those four the scalar reduction already competes and no hand-vectorized
// kernel is shipped: Go's loong64/ppc64le assemblers expose no vector double
// arithmetic (the same wall go-fft documented), riscv64's V extension is
// optional, and the per-arch qemu jobs still exercise this dispatch.

// HaveReduceSIMD reports whether this build routes Sum through a hand-vectorized
// SIMD kernel (true on amd64/arm64) rather than the scalar oracle (false on
// loong64/ppc64le/riscv64/s390x). The kernels test logs it so each per-arch CI
// run states which path it validated.
const HaveReduceSIMD = false

func sumSIMD(a []float64) float64 { return Sum(a) }
