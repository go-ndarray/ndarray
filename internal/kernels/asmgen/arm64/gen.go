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
	f.Add(sqrtKernel())
	f.Add(addKernel())
	f.Add(subKernel())
	f.Add(mulKernel())
	f.Add(gemmKernel())
	if err := os.WriteFile("sum_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_arm64.s")
}

// binSig is the elementwise binary-op signature
//
//	NAME(dst, a, b *float64, n int)
func binSig() arm64.Signature {
	return arm64.Layout(
		[]string{"dst", "a", "b", "n"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
}

// Go's arm64 assembler exposes NO plain vector double ADD/SUB/MUL/DIV — only the
// fused multiply-accumulates VFMLA/VFMLS (the same wall the sum reduction hit).
// So the elementwise add/sub/mul kernels reach the op through an FMA against a
// constant vector, each of which rounds bit-identically to the plain op:
//
//	add:  dst = b + a*1.0   via VFMLA (a, 1.0, acc=b)   — FMA(a,1,b) == a+b
//	sub:  dst = a - b*1.0   via VFMLS (b, 1.0, acc=a)   — FMA(-b,1,a) == a-b
//	mul:  dst = 0 + a*b     via VFMLA (a, b,   acc=0)   — FMA(a,b,0)  == a*b
//
// In each case the FMA's extra addend is exact (a 1.0 multiply, or a +0.0 add),
// so the single IEEE rounding lands on exactly the value the plain operation
// would, making these bit-identical to the scalar Add/Sub/Mul oracle (validated
// per-arch in CI). Division has neither an FMA form nor a vector-double divide on
// arm64, so Div stays on the scalar FDIVD oracle (divBin in reduce_arm64.go).
//
// VFMLA Va, Vb, Vacc computes Vacc += Va*Vb; VFMLS computes Vacc -= Va*Vb (the
// Go/asmgen operand order). The loop does 8 doubles (64 bytes) per iteration
// across four D2 lanes, then a scalar FP tail for the (n mod 8) remainder.

// sqrtKernel builds sqrtNEON(dst, src *float64, n int): dst[i] = sqrt(src[i])
// with the packed NEON double square root FSQRT Vd.2D (2 doubles/op, unrolled 4x
// = 8 lanes/iter) plus a scalar FSQRTD tail.
//
// Go's arm64 assembler exposes only the *scalar* FSQRTD — there is no VFSQRT
// mnemonic for the vector form — so the vector instruction is emitted directly as
// its fixed 32-bit encoding via WORD. FSQRT Vd.2D, Vn.2D is the Advanced-SIMD
// two-register-misc fp64 square root: 0110_1110_1110_0001_1111_10nn_nnnd_dddd,
// i.e. base 0x6EE1F800 | (Vn<<5) | Vd. This is the same hardware NEON double
// sqrt OpenBLAS/numpy use; it is correctly-rounded IEEE-754, identical lane-by-
// lane to the scalar FSQRTD (and to math.Sqrt), so the kernel is bit-identical to
// the scalar sqrt oracle including negatives->NaN, ±Inf, and signed zeros
// (validated bit-for-bit per-arch in CI). Using raw WORD keeps this pure Go asm
// (no cgo); it is the one instruction the assembler cannot name.
//
// This replaces the previous arm64 sqrt path (the compiler's scalar FSQRTD loop,
// one lane at a time): the packed form does two lanes per instruction, which is
// what let SqrtInto reach parity with numpy's vectorized sqrt at small n.
func sqrtKernel() *emit.Function {
	sig := arm64.Layout(
		[]string{"dst", "src", "n"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
	b := arm64.NewFunc("sqrtNEON", sig, 0)
	b.LoadArg("dst", "R0").
		LoadArg("src", "R1").
		LoadArg("n", "R2").
		// Main loop: 8 doubles (64 bytes) per iteration, packed FSQRT V0..V3.
		Raw("block:").
		Raw("CMP $8, R2").
		Raw("BLT tail").
		Raw("VLD1.P 64(R1), [V0.D2, V1.D2, V2.D2, V3.D2]").
		Raw("WORD $0x6EE1F800"). // FSQRT V0.2D, V0.2D
		Raw("WORD $0x6EE1F821"). // FSQRT V1.2D, V1.2D
		Raw("WORD $0x6EE1F842"). // FSQRT V2.2D, V2.2D
		Raw("WORD $0x6EE1F863"). // FSQRT V3.2D, V3.2D
		Raw("VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R0)").
		Raw("SUB $8, R2").
		Raw("B block").
		// Scalar tail: remaining (n mod 8) elements via scalar FSQRTD.
		Raw("tail:").
		Raw("CBZ R2, done").
		Raw("FMOVD.P 8(R1), F0").
		Raw("FSQRTD F0, F0").
		Raw("FMOVD.P F0, 8(R0)").
		Raw("SUB $1, R2").
		Raw("B tail").
		Raw("done:").
		Ret()
	return b.Func()
}

// onesVec emits V31 = [1.0, 1.0] (the IEEE bit pattern of 1.0 broadcast).
func onesVec(b *arm64.Builder) {
	b.Raw("MOVD $0x3FF0000000000000, R7")
	b.Raw("VDUP R7, V31.D2")
}

// addKernel builds addNEON(dst, a, b *float64, n int): dst = a + b via FMA.
func addKernel() *emit.Function {
	b := arm64.NewFunc("addNEON", binSig(), 0)
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("n", "R3")
	onesVec(b)
	b.Raw("block:").Raw("CMP $8, R3").Raw("BLT tail").
		Raw("VLD1.P 64(R2), [V0.D2, V1.D2, V2.D2, V3.D2]"). // acc = b
		Raw("VLD1.P 64(R1), [V4.D2, V5.D2, V6.D2, V7.D2]"). // a
		Raw("VFMLA V4.D2, V31.D2, V0.D2").                  // acc += a*1.0 => a+b
		Raw("VFMLA V5.D2, V31.D2, V1.D2").
		Raw("VFMLA V6.D2, V31.D2, V2.D2").
		Raw("VFMLA V7.D2, V31.D2, V3.D2").
		Raw("VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R0)").
		Raw("SUB $8, R3").Raw("B block").
		Raw("tail:").Raw("CBZ R3, done").
		Raw("FMOVD.P 8(R1), F0").Raw("FMOVD.P 8(R2), F1").
		Raw("FADDD F1, F0, F0").
		Raw("FMOVD.P F0, 8(R0)").
		Raw("SUB $1, R3").Raw("B tail").
		Raw("done:").Ret()
	return b.Func()
}

// subKernel builds subNEON(dst, a, b *float64, n int): dst = a - b via FMA.
func subKernel() *emit.Function {
	b := arm64.NewFunc("subNEON", binSig(), 0)
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("n", "R3")
	onesVec(b)
	b.Raw("block:").Raw("CMP $8, R3").Raw("BLT tail").
		Raw("VLD1.P 64(R1), [V0.D2, V1.D2, V2.D2, V3.D2]"). // acc = a
		Raw("VLD1.P 64(R2), [V4.D2, V5.D2, V6.D2, V7.D2]"). // b
		Raw("VFMLS V4.D2, V31.D2, V0.D2").                  // acc -= b*1.0 => a-b
		Raw("VFMLS V5.D2, V31.D2, V1.D2").
		Raw("VFMLS V6.D2, V31.D2, V2.D2").
		Raw("VFMLS V7.D2, V31.D2, V3.D2").
		Raw("VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R0)").
		Raw("SUB $8, R3").Raw("B block").
		Raw("tail:").Raw("CBZ R3, done").
		Raw("FMOVD.P 8(R1), F0").Raw("FMOVD.P 8(R2), F1").
		Raw("FSUBD F1, F0, F0").
		Raw("FMOVD.P F0, 8(R0)").
		Raw("SUB $1, R3").Raw("B tail").
		Raw("done:").Ret()
	return b.Func()
}

// mulKernel builds mulNEON(dst, a, b *float64, n int): dst = a * b via FMA into a
// -0.0 accumulator: FMA(a, b, -0.0) == a*b *bit-for-bit, including sign-of-zero*.
// Adding +0.0 would NOT be safe — when the product is -0.0, (-0.0)+(+0.0) rounds
// to +0.0 under round-to-nearest, flipping the sign that plain a*b keeps. Adding
// -0.0 is the correct identity: -0.0+(-0.0) = -0.0 and (+0.0)+(-0.0) = +0.0, so
// the result sign matches the product's, and for any nonzero product adding -0.0
// changes nothing. Hence the tile is bit-identical to the scalar Mul oracle
// (asserted in TestBinSIMD, which pairs -0.0*0.0 etc.).
func mulKernel() *emit.Function {
	b := arm64.NewFunc("mulNEON", binSig(), 0)
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("n", "R3")
	// V30 = [-0.0, -0.0] (IEEE bits 0x8000000000000000), the FMA accumulator seed.
	b.Raw("MOVD $0x8000000000000000, R7")
	b.Raw("VDUP R7, V30.D2")
	b.Raw("block:").Raw("CMP $8, R3").Raw("BLT tail").
		Raw("VLD1.P 64(R1), [V0.D2, V1.D2, V2.D2, V3.D2]"). // a
		Raw("VLD1.P 64(R2), [V4.D2, V5.D2, V6.D2, V7.D2]"). // b
		Raw("VMOV V30.B16, V8.B16").                        // acc = -0.0
		Raw("VMOV V30.B16, V9.B16").
		Raw("VMOV V30.B16, V10.B16").
		Raw("VMOV V30.B16, V11.B16").
		Raw("VFMLA V0.D2, V4.D2, V8.D2"). // acc += a*b => a*b (sign-exact)
		Raw("VFMLA V1.D2, V5.D2, V9.D2").
		Raw("VFMLA V2.D2, V6.D2, V10.D2").
		Raw("VFMLA V3.D2, V7.D2, V11.D2").
		Raw("VST1.P [V8.D2, V9.D2, V10.D2, V11.D2], 64(R0)").
		Raw("SUB $8, R3").Raw("B block").
		Raw("tail:").Raw("CBZ R3, done").
		Raw("FMOVD.P 8(R1), F0").Raw("FMOVD.P 8(R2), F1").
		Raw("FMULD F1, F0, F0").
		Raw("FMOVD.P F0, 8(R0)").
		Raw("SUB $1, R3").Raw("B tail").
		Raw("done:").Ret()
	return b.Func()
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

// fmlaElem returns the Plan 9 line for the by-element (indexed) double FMLA
//
//	FMLA V{acc}.2D, V{breg}.2D, V{areg}.D[idx]   =>  V{acc} += V{breg} * V{areg}[idx]
//
// which Go's arm64 assembler cannot name, emitted as its raw 32-bit encoding via
// WORD. The encoding (Advanced SIMD, FMLA by element, FP64, ARM C7-2: Q=1, sz=1,
// L=0) is
//
//	0|1|0|01111|1|1|0|M|Rm[3:0]|0001|H|0|Rn|Rd
//
// with the element index = H (sz=1 so L is unused) and Rm = M:Rm[3:0]. The result
// includes a trailing comment naming the human-readable instruction so the
// generated .s stays auditable; the encoding is cross-checked against objdump in
// the asmgen tests. areg/breg/acc are V-register numbers, idx in {0,1}.
func fmlaElem(acc, breg, areg, idx int) string {
	const base = 0x4FC01000 // Q=1, sz=1, opcode 0001, base for FMLA-by-elem .2D
	M := (areg >> 4) & 1
	Rm := areg & 0xF
	H := idx & 1
	word := base | (M << 20) | (Rm << 16) | (H << 11) | (breg << 5) | acc
	return fmt.Sprintf("WORD $0x%08X // FMLA V%d.2D, V%d.2D, V%d.D[%d]",
		word, acc, breg, areg, idx)
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
// the per-step B row in V16..V19 (8 doubles), the four A values in V20,V21 (two
// D2 vectors, two A doubles each), and a vector of 1.0 in V31 for the final
// C += acc.
//
// Two arm64-assembler constraints shape the kernel (both noted at point of use):
//  1. THE BY-LANE FMLA. Every hand-tuned arm64 dgemm broadcasts one A lane
//     straight into the FMA with FMLA Vd.2d, Vn.2d, Vm.d[i]. Go's arm64 assembler
//     cannot *name* this indexed-element form ("illegal combination ... ELEM"),
//     so it is emitted by its raw WORD encoding (fmlaElem below: cross-checked
//     against objdump). The four packed A doubles of a k-step are loaded as two
//     D2 vectors (V20=[a0,a1], V21=[a2,a3]) and each row's FMA reads its A scalar
//     by lane — eliminating the MOVD-to-GP + VDUP round-trip the prior kernel
//     paid per A value every k-step. Measured on this Apple-silicon core this is
//     ~1.28x faster on the L1-resident micro-kernel (44 -> 58 GFLOP/s/core);
//     the earlier note claiming the indexed form is "slower" was wrong (it was
//     never actually measured against a correct encoding). See docs/perf.md.
//  2. There is no plain vector FP add (only VFMLA/VFMLS), so the closing C += acc
//     is done as C = C + acc*1.0 via VFMLA against V31=[1.0,1.0]. Multiplying by
//     exactly 1.0 is exact and FMA(acc,1.0,C) rounds identically to C+acc (C is
//     the only inexact addend), so this is a true add — the tile result is
//     bit-identical to the scalar oracle's ikj accumulation order.
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
		// A col: 4 contiguous doubles into two D2 vectors V20=[a0,a1], V21=[a2,a3].
		// Each row's FMA reads its A scalar by lane (header note 1) — no VDUP.
		Raw("VLD1.P 32(R1), [V20.D2, V21.D2]").
		// FMLA Vacc.2D, Vb.2D, Va.D[i] => Vacc += Vb * Va[i].  row r uses A lane
		// (r&1) of V20 (r<2) or V21 (r>=2).
		Raw(fmlaElem(0, 16, 20, 0)). // row0 = a0
		Raw(fmlaElem(1, 17, 20, 0)).
		Raw(fmlaElem(2, 18, 20, 0)).
		Raw(fmlaElem(3, 19, 20, 0)).
		Raw(fmlaElem(4, 16, 20, 1)). // row1 = a1
		Raw(fmlaElem(5, 17, 20, 1)).
		Raw(fmlaElem(6, 18, 20, 1)).
		Raw(fmlaElem(7, 19, 20, 1)).
		Raw(fmlaElem(8, 16, 21, 0)). // row2 = a2
		Raw(fmlaElem(9, 17, 21, 0)).
		Raw(fmlaElem(10, 18, 21, 0)).
		Raw(fmlaElem(11, 19, 21, 0)).
		Raw(fmlaElem(12, 16, 21, 1)). // row3 = a3
		Raw(fmlaElem(13, 17, 21, 1)).
		Raw(fmlaElem(14, 18, 21, 1)).
		Raw(fmlaElem(15, 19, 21, 1)).
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
