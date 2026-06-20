//go:build ignore

// Command gen produces sum_arm64.s, the NEON float64 sum-reduction kernel, via
// go-asmgen. Run with: go run gen.go (or `go generate` from the kernels
// package).
//
// NumPy's sum ufunc is single-threaded but SIMD-vectorized C with several
// accumulator lanes; Go's scalar reduction loop is a single dependency chain
// (`s += v`) the gc compiler will not auto-vectorize, because that would change
// floating-point grouping. So we hand-vectorize with NEON and multiple
// accumulators, the same parallelism numpy uses inside one core.
//
//   sumNEON(a *float64, n int) float64
//     Sums a[0..n) with FOUR independent V.D2 accumulators (8 float64 lanes),
//     then folds the lanes pairwise and adds a scalar tail. Floating-point
//     addition is not associative, so this lane-parallel grouping can differ
//     from a strictly sequential sum by a few ULP — exactly the trade-off
//     numpy's own pairwise/SIMD summation makes. It is therefore NOT held
//     bit-identical to the scalar oracle; the kernels package validates it to a
//     tight relative tolerance against the oracle and against numpy.
//
// (A max-reduction SIMD kernel is intentionally not shipped: NEON FMAX's NaN
// propagation differs from the scalar `if v > m` oracle, so a bit-identical
// vector max is not expressible without a NaN pre-scan; Max stays on the
// associative scalar parallel path.)
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func reduceSig() arm64.Signature {
	return arm64.Layout(
		[]string{"a", "n"}, []arm64.Type{arm64.Ptr, arm64.Int64},
		[]string{"ret"}, []arm64.Type{arm64.Float64},
	)
}

func main() {
	f := emit.NewFile("arm64")
	f.Add(sumKernel())
	if err := os.WriteFile("sum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_arm64.s")
}

// sumKernel builds sumNEON(a *float64, n int) float64.
//
// Go's arm64 assembler exposes no plain vector double ADD (only VFMLA/VFMLS,
// the fused multiply-accumulates). So the lane-parallel accumulation is done
// with VFMLA against a vector of 1.0: acc += src * 1.0. Multiplying a double by
// exactly 1.0 is exact, and FMA(src, 1.0, acc) rounds identically to acc+src
// (the addend is the only inexact step), so this is a true vector add, not an
// approximation — it simply reaches it through the only vector-FP op available.
func sumKernel() *emit.Function {
	b := arm64.NewFunc("sumNEON", reduceSig(), 0)
	b.LoadArg("a", "R0").
		LoadArg("n", "R1").
		// Build the all-ones multiplier vector V16 = [1.0, 1.0] by broadcasting
		// the IEEE-754 bit pattern of 1.0 from a general register (Go's arm64
		// VDUP duplicates from an R register, not an F register).
		Raw("MOVD $0x3FF0000000000000, R2"). // bits of 1.0
		Raw("VDUP R2, V16.D2").
		// Zero the four accumulators V0..V3 (8 lanes).
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("VEOR V2.B16, V2.B16, V2.B16").
		Raw("VEOR V3.B16, V3.B16, V3.B16").
		// Main loop: 8 float64 (64 bytes) per iteration.
		Raw("block:").
		Raw("CMP $8, R1").
		Raw("BLT fold").
		Raw("VLD1.P 64(R0), [V4.D2, V5.D2, V6.D2, V7.D2]").
		Raw("VFMLA V4.D2, V16.D2, V0.D2"). // V0 += V4 * 1.0
		Raw("VFMLA V5.D2, V16.D2, V1.D2").
		Raw("VFMLA V6.D2, V16.D2, V2.D2").
		Raw("VFMLA V7.D2, V16.D2, V3.D2").
		Raw("SUB $8, R1").
		Raw("B block").
		// Fold the four vector accumulators into V0, then its two lanes into F0.
		Raw("fold:").
		Raw("VFMLA V1.D2, V16.D2, V0.D2").
		Raw("VFMLA V3.D2, V16.D2, V2.D2").
		Raw("VFMLA V2.D2, V16.D2, V0.D2").
		// Fold V0's two double lanes: extract each to a GP register (Go arm64
		// VMOV lane -> R, not -> F), move to F regs, and add.
		Raw("VMOV V0.D[0], R3").
		Raw("VMOV V0.D[1], R4").
		Raw("FMOVD R3, F0").
		Raw("FMOVD R4, F9").
		Raw("FADDD F9, F0, F0"). // F0 = lane0 + lane1
		// Scalar tail: remaining (n mod 8) elements.
		Raw("tail:").
		Raw("CBZ R1, done").
		Raw("FMOVD.P 8(R0), F10").
		Raw("FADDD F10, F0, F0").
		Raw("SUB $1, R1").
		Raw("B tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
