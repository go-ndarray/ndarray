package kernels

import "github.com/go-simd/floats"

// sumSIMD returns the sum of a (empty -> 0) in NumPy's order: floats.SumPairwise,
// bit-identical to np.sum of a contiguous 1-D float64 array on every architecture
// (K1 of the go-scikit performance campaign: the kernel lives in go-simd/floats).
//
// This replaces the lane-parallel sums of the per-architecture kernels, whose
// grouping differed from NumPy's (and from one architecture to the next) by a few
// ULP. NumPy's order is the reference, so a changed last bit is the fix.
func sumSIMD(a []float64) float64 { return floats.SumPairwise(a) }
