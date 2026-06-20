//go:build !arm64 && !amd64

package kernels

// On architectures without a generated SIMD kernel, the dispatch wrappers route
// to the scalar oracle. sumSIMD/sqrtSIMD/maxSIMD/minSIMD alias the oracles so the
// SIMD-vs-scalar tests (which then trivially hold, scalar == scalar) still
// exercise this path and the package builds uniformly across all six targets.
//
// amd64 (SSE2) and arm64 (NEON sum + intrinsic scalar sqrt/max/min) ship kernels
// (see reduce_amd64.go / reduce_arm64.go), so this file is the
// loong64/ppc64le/riscv64/s390x fallback. For those four no hand-vectorized
// kernel is shipped: Go's loong64/ppc64le assemblers expose no vector-double
// arithmetic (the same wall go-fft documented), riscv64's V extension is
// optional, and the per-arch qemu jobs still exercise this dispatch. The scalar
// math.Sqrt/math.Max/math.Min still lower to the hardware sqrt/max where the
// target has it, and the multicore path beats single-threaded numpy on large
// arrays regardless.

// HaveReduceSIMD reports whether this build routes through a hand-vectorized
// SIMD kernel (true on amd64/arm64) rather than the scalar oracle (false on
// loong64/ppc64le/riscv64/s390x). The kernels test logs it so each per-arch CI
// run states which path it validated.
const HaveReduceSIMD = false

func sumSIMD(a []float64) float64 { return Sum(a) }
func sqrtSIMD(dst, src []float64) { sqrtScalar(dst, src) }
func maxSIMD(a []float64) float64 { return maxUnrolled(a) }
func minSIMD(a []float64) float64 { return minUnrolled(a) }
