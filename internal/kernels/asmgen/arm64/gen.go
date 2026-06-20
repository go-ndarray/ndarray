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
	f.Add(gemmKernel())
	if err := os.WriteFile("sum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_arm64.s")
}

// gemmSig is the micro-kernel signature
//
//	gemmMicro4x8(kc int, pa, pb, c *float64, ldc int)
func gemmSig() arm64.Signature {
	return arm64.Layout(
		[]string{"kc", "pa", "pb", "c", "ldc"},
		[]arm64.Type{arm64.Int64, arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
}

// gemmKernel builds gemmMicro4x8(kc int, pa, pb, c *float64, ldc int): the NEON
// 4x8 register-blocked GEMM micro-kernel at the heart of the packed GEMM.
//
// It computes a 4-row x 8-col tile of C from packed panels and ADDS it into C:
//
//	for p in [0,kc):  C[r][col] += pa[p*4+r] * pb[p*8+col]   (r<4, col<8)
//
// pa is the packed A panel (MR=4 contiguous A values per k step), pb the packed
// B panel (NR=8 contiguous B values per k step) — both unit-stride, so the loads
// are conflict-free regardless of the source matrices' strides. This contiguity
// is the whole point of packing: it removes the power-of-two L1 set-conflicts
// that defeated the earlier un-packed register-blocked attempt.
//
// Register map: the 4x8 C tile lives in V0..V15 (4 rows x 4 D2 = 16 accumulators);
// the per-step B row in V16..V19 (8 doubles), the four broadcast A values in
// V20..V23, and a vector of 1.0 in V31 for the final C += acc.
//
// Two arm64-assembler constraints shape the kernel (both noted at point of use):
//  1. Go's arm64 assembler has no by-lane FMLA (FMLA Vd.2d, Vn.2d, Vm.d[i]), the
//     instruction every hand-tuned dgemm uses to broadcast one A lane. So each A
//     scalar is moved to a GP register and VDUP'd into a full D2 vector, then a
//     plain vector-by-vector VFMLA is issued. The dup is amortized across all 8
//     B columns of the row.
//  2. There is no plain vector FP add either (only VFMLA/VFMLS), so the closing
//     C += acc is done as C = C + acc*1.0 via VFMLA against V31=[1.0,1.0].
//     Multiplying by exactly 1.0 is exact and FMA(acc,1.0,C) rounds identically
//     to C+acc (C is the only inexact addend), so this is a true add — the tile
//     result is bit-identical to the scalar oracle's ikj accumulation order.
func gemmKernel() *emit.Function {
	b := arm64.NewFunc("gemmMicro4x8", gemmSig(), 0)
	b.LoadArg("kc", "R0").
		LoadArg("pa", "R1").
		LoadArg("pb", "R2").
		LoadArg("c", "R3").
		LoadArg("ldc", "R4").
		Raw("LSL $3, R4, R4"). // ldc -> bytes
		// Zero the sixteen C accumulators V0..V15.
		Raw("VEOR V0.B16, V0.B16, V0.B16").
		Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("VEOR V2.B16, V2.B16, V2.B16").
		Raw("VEOR V3.B16, V3.B16, V3.B16").
		Raw("VEOR V4.B16, V4.B16, V4.B16").
		Raw("VEOR V5.B16, V5.B16, V5.B16").
		Raw("VEOR V6.B16, V6.B16, V6.B16").
		Raw("VEOR V7.B16, V7.B16, V7.B16").
		Raw("VEOR V8.B16, V8.B16, V8.B16").
		Raw("VEOR V9.B16, V9.B16, V9.B16").
		Raw("VEOR V10.B16, V10.B16, V10.B16").
		Raw("VEOR V11.B16, V11.B16, V11.B16").
		Raw("VEOR V12.B16, V12.B16, V12.B16").
		Raw("VEOR V13.B16, V13.B16, V13.B16").
		Raw("VEOR V14.B16, V14.B16, V14.B16").
		Raw("VEOR V15.B16, V15.B16, V15.B16").
		Raw("CBZ R0, gstore").
		Raw("gkloop:").
		// B row: 8 contiguous doubles into V16..V19.
		Raw("VLD1.P 64(R2), [V16.D2, V17.D2, V18.D2, V19.D2]").
		// A col: 4 contiguous doubles, each broadcast to a D2 vector (no by-lane
		// FMLA available — see header note 1).
		Raw("MOVD.P 8(R1), R7").
		Raw("VDUP R7, V20.D2").
		Raw("MOVD.P 8(R1), R7").
		Raw("VDUP R7, V21.D2").
		Raw("MOVD.P 8(R1), R7").
		Raw("VDUP R7, V22.D2").
		Raw("MOVD.P 8(R1), R7").
		Raw("VDUP R7, V23.D2").
		// VFMLA Va, Vb, Vacc => Vacc += Va*Vb.  Cacc += Brow * Abroadcast.
		Raw("VFMLA V16.D2, V20.D2, V0.D2"). // row0
		Raw("VFMLA V17.D2, V20.D2, V1.D2").
		Raw("VFMLA V18.D2, V20.D2, V2.D2").
		Raw("VFMLA V19.D2, V20.D2, V3.D2").
		Raw("VFMLA V16.D2, V21.D2, V4.D2"). // row1
		Raw("VFMLA V17.D2, V21.D2, V5.D2").
		Raw("VFMLA V18.D2, V21.D2, V6.D2").
		Raw("VFMLA V19.D2, V21.D2, V7.D2").
		Raw("VFMLA V16.D2, V22.D2, V8.D2"). // row2
		Raw("VFMLA V17.D2, V22.D2, V9.D2").
		Raw("VFMLA V18.D2, V22.D2, V10.D2").
		Raw("VFMLA V19.D2, V22.D2, V11.D2").
		Raw("VFMLA V16.D2, V23.D2, V12.D2"). // row3
		Raw("VFMLA V17.D2, V23.D2, V13.D2").
		Raw("VFMLA V18.D2, V23.D2, V14.D2").
		Raw("VFMLA V19.D2, V23.D2, V15.D2").
		Raw("SUB $1, R0").
		Raw("CBNZ R0, gkloop").
		// C += acc, row by row, via VFMLA against V31=[1.0,1.0] (no vector FP add
		// available — see header note 2). Bit-identical to C+acc.
		Raw("gstore:").
		Raw("MOVD $0x3FF0000000000000, R6").
		Raw("VDUP R6, V31.D2").
		Raw("MOVD R3, R5"). // row0
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V0.D2, V31.D2, V16.D2").
		Raw("VFMLA V1.D2, V31.D2, V17.D2").
		Raw("VFMLA V2.D2, V31.D2, V18.D2").
		Raw("VFMLA V3.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row1
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V4.D2, V31.D2, V16.D2").
		Raw("VFMLA V5.D2, V31.D2, V17.D2").
		Raw("VFMLA V6.D2, V31.D2, V18.D2").
		Raw("VFMLA V7.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row2
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V8.D2, V31.D2, V16.D2").
		Raw("VFMLA V9.D2, V31.D2, V17.D2").
		Raw("VFMLA V10.D2, V31.D2, V18.D2").
		Raw("VFMLA V11.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Raw("ADD R4, R3, R3"). // row3
		Raw("MOVD R3, R5").
		Raw("VLD1 (R5), [V16.D2, V17.D2, V18.D2, V19.D2]").
		Raw("VFMLA V12.D2, V31.D2, V16.D2").
		Raw("VFMLA V13.D2, V31.D2, V17.D2").
		Raw("VFMLA V14.D2, V31.D2, V18.D2").
		Raw("VFMLA V15.D2, V31.D2, V19.D2").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R5)").
		Ret()
	return b.Func()
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
