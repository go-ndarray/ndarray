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
//	sumNEON(a *float64, n int) float64
//	  Sums a[0..n) with FOUR independent V.D2 accumulators (8 float64 lanes),
//	  then folds the lanes pairwise and adds a scalar tail. Floating-point
//	  addition is not associative, so this lane-parallel grouping can differ
//	  from a strictly sequential sum by a few ULP — exactly the trade-off
//	  numpy's own pairwise/SIMD summation makes. It is therefore NOT held
//	  bit-identical to the scalar oracle; the kernels package validates it to a
//	  tight relative tolerance against the oracle and against numpy.
//
// (No max-reduction kernel is generated here: Max and Min come from
// go-simd/floats, whose NEON extrema kernels (extrema_arm64.s there) use
// FMAX/FMIN, which propagate NaN and order -0 below +0 as Go's max/min do.)
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
	// The float64 sum, sqrt, add/sub/mul/div and max/min kernels moved to
	// go-simd/floats (K1 of the go-scikit performance campaign): sumKernel,
	// sqrtKernel, binKernel and extremumKernel below are no longer emitted.
	// max32NEON/min32NEON moved to go-simd/floats (Float32Max, Float32Min;
	// go-ndarray/ndarray#98).
	// The GEMM kernels moved to github.com/go-blas/blas, which generates them.
	f.Add(blocks8Kernel())
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

// The elementwise kernels use the plain vector double ops VFADD/VFSUB/VFMUL/
// VFDIV, which Go's arm64 assembler names since Go 1.27 (this module requires
// 1.27.1). Before that only the fused VFMLA/VFMLS existed, and add/sub/mul went
// through an FMA against an exact constant (b + a*1.0, a - b*1.0, -0.0 + a*b),
// bit-identical but harder to read, while division stayed scalar. Each lane is
// one correctly rounded IEEE-754 operation, the same the scalar FADDD/FSUBD/
// FMULD/FDIVD perform, so the kernels are bit-identical to the scalar oracle
// (TestBinSIMD). Go's operand order is VFSUB Vm, Vn, Vd: Vd = Vn - Vm.
//
// The loop does 8 doubles (64 bytes) per iteration across four D2 lanes, then a
// scalar FP tail for the (n mod 8) remainder.

// sqrtKernel builds sqrtNEON(dst, src *float64, n int): dst[i] = sqrt(src[i])
// with the packed NEON double square root FSQRT Vd.2D (2 doubles/op, unrolled 4x
// = 8 lanes/iter) plus a scalar FSQRTD tail.
//
// FSQRT Vd.2D is VFSQRT Vn.D2, Vd.D2 in Go since 1.27; it was emitted as a raw
// WORD before (0x6EE1F800 | Vn<<5 | Vd, the same encoding, checked against Go's
// own disassembly). It is correctly rounded IEEE-754, identical lane by lane to
// the scalar FSQRTD and math.Sqrt, so the kernel is bit-identical to the scalar
// oracle including negatives->NaN, ±Inf and signed zeros.
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
		Raw("VFSQRT V0.D2, V0.D2").
		Raw("VFSQRT V1.D2, V1.D2").
		Raw("VFSQRT V2.D2, V2.D2").
		Raw("VFSQRT V3.D2, V3.D2").
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

// binKernel builds NAME(dst, a, b *float64, n int): dst[i] = a[i] OP b[i],
// with vop the vector form (Go operand order VOP Vm, Vn, Vd: Vd = Vn OP Vm) and
// sop the scalar one for the tail (SOP Fm, Fn, Fd: Fd = Fn OP Fm).
func binKernel(name, vop, sop string) *emit.Function {
	b := arm64.NewFunc(name, binSig(), 0)
	b.LoadArg("dst", "R0").LoadArg("a", "R1").LoadArg("b", "R2").LoadArg("n", "R3")
	b.Raw("block:").Raw("CMP $8, R3").Raw("BLT tail").
		Raw("VLD1.P 64(R1), [V0.D2, V1.D2, V2.D2, V3.D2]"). // a
		Raw("VLD1.P 64(R2), [V4.D2, V5.D2, V6.D2, V7.D2]")  // b
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("%s V%d.D2, V%d.D2, V%d.D2", vop, 4+r, r, r)) // a OP b
	}
	b.Raw("VST1.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R0)").
		Raw("SUB $8, R3").Raw("B block").
		Raw("tail:").Raw("CBZ R3, done").
		Raw("FMOVD.P 8(R1), F0").Raw("FMOVD.P 8(R2), F1").
		Raw(sop + " F1, F0, F0").
		Raw("FMOVD.P F0, 8(R0)").
		Raw("SUB $1, R3").Raw("B tail").
		Raw("done:").Ret()
	return b.Func()
}

// blocks8Kernel builds blocks8NEON(r, a *float32, blocks int): for each of
// the blocks of eight float32 at a, in order, r[j] += a[8i+j]. The eight
// accumulators live in V0 (r[0:4]) and V1 (r[4:8]); VFADD on .S4 is one IEEE
// float32 addition per lane, so the result is the scalar loop's, bit for bit
// (NumPy's pairwise-sum block, whose order it keeps). blocks >= 1.
func blocks8Kernel() *emit.Function {
	sig := arm64.Layout(
		[]string{"r", "a", "blocks"},
		[]arm64.Type{arm64.Ptr, arm64.Ptr, arm64.Int64},
		nil, nil,
	)
	b := arm64.NewFunc("blocks8NEON", sig, 0)
	b.LoadArg("r", "R0").LoadArg("a", "R1").LoadArg("blocks", "R2").
		Raw("VLD1 (R0), [V0.S4, V1.S4]").
		Raw("b8loop:").
		Raw("VLD1.P 32(R1), [V2.S4, V3.S4]").
		Raw("VFADD V2.S4, V0.S4, V0.S4").
		Raw("VFADD V3.S4, V1.S4, V1.S4").
		Raw("SUB $1, R2").
		Raw("CBNZ R2, b8loop").
		Raw("VST1 [V0.S4, V1.S4], (R0)").
		Ret()
	return b.Func()
}

// extremumKernel builds NAME(a *float64, n int) float64, the maximum (or
// minimum) of a, for n >= 8: four D2 accumulators seeded with the first eight
// elements, folded together, then the two lanes, then the tail one at a time.
// FMAX/FMIN return NaN when either operand is NaN and order -0 below +0, as
// the scalar FMAXD/FMIND that Go's builtin max/min compile to do, and the
// operation is associative under those rules, so the result is bit-identical
// to the serial oracle (TestMaxMinSIMD).
func extremumKernel(name, vop, sop string) *emit.Function {
	b := arm64.NewFunc(name, reduceSig(), 0)
	b.LoadArg("a", "R0").LoadArg("n", "R1").
		Raw("VLD1.P 64(R0), [V0.D2, V1.D2, V2.D2, V3.D2]").
		Raw("SUB $8, R1").
		Raw("block:").Raw("CMP $8, R1").Raw("BLT fold").
		Raw("VLD1.P 64(R0), [V4.D2, V5.D2, V6.D2, V7.D2]")
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("%s V%d.D2, V%d.D2, V%d.D2", vop, 4+r, r, r))
	}
	b.Raw("SUB $8, R1").Raw("B block").
		Raw("fold:").
		Raw(vop+" V1.D2, V0.D2, V0.D2").
		Raw(vop+" V3.D2, V2.D2, V2.D2").
		Raw(vop+" V2.D2, V0.D2, V0.D2").
		Raw("VMOV V0.D[0], R3").
		Raw("VMOV V0.D[1], R4").
		Raw("FMOVD R3, F0").
		Raw("FMOVD R4, F9").
		Raw(sop+" F9, F0, F0").
		Raw("tail:").Raw("CBZ R1, done").
		Raw("FMOVD.P 8(R0), F10").
		Raw(sop+" F10, F0, F0").
		Raw("SUB $1, R1").Raw("B tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}

// sumKernel builds sumNEON(a *float64, n int) float64.
//
// The lane-parallel accumulation is VFMLA against a vector of 1.0: acc += src *
// 1.0 (written when Go's assembler had no VFADD; it has since Go 1.27). Multiplying a double by
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
