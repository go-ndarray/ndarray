package kernels

import "github.com/go-simd/floats"

// dotSIMD is the inner product of two equal-length slices: go-simd/floats.Dot,
// the one shared kernel (AVX2/FMA, NEON, LASX, VSX, RVV, VXE, or its portable
// loop). It panics when the lengths differ. The sum is reassociated across the
// kernel's lanes and fused, so the result is within gamma_n * sum|a_i b_i| of
// the exact value, not bit-equal to a left-to-right loop.
func dotSIMD(a, b []float64) float64 { return floats.Dot(a, b) }
