// Command gen produces sum_amd64.s, the SSE2 float64 kernels, via go-asmgen.
// Run with: go run gen.go (or `go generate` from the kernels package).
//
//	sumSSE2(a *float64, n int) float64
//	  Sums a[0..n) with FOUR independent XMM accumulators (8 float64 lanes via
//	  2 doubles per register), then folds them and adds a scalar tail. SSE2 is
//	  part of the amd64 baseline (GOAMD64=v1), so the kernel is always callable
//	  with no CPU-feature branch. The lane-parallel grouping is a
//	  different-but-valid summation order, so it is validated to a tight relative
//	  tolerance against the scalar oracle (and numpy), not held bit-identical.
//
//	sqrtSSE2(dst, src *float64, n int)
//	  Writes sqrt(src[i]) into dst[i] with packed SQRTPD (2 doubles per op,
//	  unrolled 4x = 8 lanes/iter) plus a scalar SQRTSD tail. SQRTPD is the same
//	  correctly-rounded IEEE square root math.Sqrt computes, so this is
//	  bit-identical to the scalar oracle including NaN/Inf/signed-zero.
//
//	maxSSE2(a *float64, n int) float64 / minSSE2(a *float64, n int) float64
//	  NaN-propagating max/min reduction (numpy.max / Go math.Max semantics): if
//	  any element is NaN the result is NaN, else the largest/smallest. Packed
//	  MAXPD/MINPD give the extreme over non-NaN data; a parallel CMPPD-unordered
//	  ($3) OR-accumulator records whether ANY lane ever held a NaN, and the fold
//	  forces the result to NaN if so. MAXPD/MINPD's own (non-propagating) NaN
//	  rule is irrelevant because the NaN mask overrides the value in that case.
//	  Accumulators are seeded by broadcasting a[0] (MOVDDUP), so n>=1 (the
//	  caller guarantees non-empty) needs no -Inf/+Inf constant. Bit-identical to
//	  the scalar Max/Min oracle.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(sumKernel())
	f.Add(sqrtKernel())
	f.Add(extremeKernel("maxSSE2", "MAXPD", "MAXSD"))
	f.Add(extremeKernel("minSSE2", "MINPD", "MINSD"))
	f.Add(binKernel("addSSE2", "ADDPD", "ADDSD"))
	f.Add(binKernel("subSSE2", "SUBPD", "SUBSD"))
	f.Add(binKernel("mulSSE2", "MULPD", "MULSD"))
	f.Add(binKernel("divSSE2", "DIVPD", "DIVSD"))
	f.Add(gemmFMAKernel())
	f.Add(gemm32FMAKernel())
	for mr := 1; mr <= gemmMR; mr++ {
		f.Add(gemm32UAKernel(mr))
		f.Add(gemmUAKernel(mr, false))
	}
	f.Add(dotFMAKernel())
	f.Add(gemmSSE2HalfKernel())
	f.Add(amd64.FeatureProbe("hasFMA", amd64.FMA))
	f.Add(blocks8Kernel())
	f.Add(bin32Kernel("add32SSE2", "ADDPS", "ADDSS"))
	f.Add(bin32Kernel("sub32SSE2", "SUBPS", "SUBSS"))
	f.Add(bin32Kernel("mul32SSE2", "MULPS", "MULSS"))
	f.Add(bin32Kernel("div32SSE2", "DIVPS", "DIVSS"))
	f.Add(sqrt32Kernel())
	f.Add(extremum32Kernel("max32SSE2", "MAXPS", "MAXSS"))
	f.Add(extremum32Kernel("min32SSE2", "MINPS", "MINSS"))

	if err := os.WriteFile("sum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_amd64.s")
}

func reduceSig() amd64.Signature {
	return amd64.Layout(
		[]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float64},
	)
}

// sumKernel builds sumSSE2(a *float64, n int) float64.
func sumKernel() *emit.Function {
	b := amd64.NewFunc("sumSSE2", reduceSig(), 0)
	b.LoadArg("a", "AX").
		LoadArg("n", "CX").
		Raw("XORPS X0, X0").
		Raw("XORPS X1, X1").
		Raw("XORPS X2, X2").
		Raw("XORPS X3, X3").
		Raw("block:").
		Raw("CMPQ CX, $8").
		Raw("JL fold").
		Raw("MOVUPD (AX), X4").
		Raw("MOVUPD 16(AX), X5").
		Raw("MOVUPD 32(AX), X6").
		Raw("MOVUPD 48(AX), X7").
		Raw("ADDPD X4, X0").
		Raw("ADDPD X5, X1").
		Raw("ADDPD X6, X2").
		Raw("ADDPD X7, X3").
		Raw("ADDQ $64, AX").
		Raw("SUBQ $8, CX").
		Raw("JMP block").
		Raw("fold:").
		Raw("ADDPD X1, X0").
		Raw("ADDPD X3, X2").
		Raw("ADDPD X2, X0").
		Raw("MOVAPS X0, X1").
		Raw("UNPCKHPD X1, X1").
		Raw("ADDSD X1, X0").
		Raw("tail:").
		Raw("TESTQ CX, CX").
		Raw("JZ done").
		Raw("ADDSD (AX), X0").
		Raw("ADDQ $8, AX").
		Raw("DECQ CX").
		Raw("JMP tail").
		Raw("done:").
		StoreRet("X0", "ret").
		Ret()
	return b.Func()
}

// blocks8Kernel builds blocks8SSE2(r, a *float32, blocks int): for each of
// the blocks of eight float32 at a, in order, r[j] += a[8i+j]. The eight
// accumulators live in X0 (r[0:4]) and X1 (r[4:8]); ADDPS is one IEEE
// float32 addition per lane, so the result is the scalar loop's, bit for bit
// (NumPy's pairwise-sum block, whose order it keeps). blocks >= 1.
func blocks8Kernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"r", "a", "blocks"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc("blocks8SSE2", sig, 0)
	b.LoadArg("r", "DI").
		LoadArg("a", "SI").
		LoadArg("blocks", "CX").
		Raw("MOVUPS (DI), X0").
		Raw("MOVUPS 16(DI), X1").
		Raw("b8loop:").
		Raw("MOVUPS (SI), X2").
		Raw("MOVUPS 16(SI), X3").
		Raw("ADDPS X2, X0").
		Raw("ADDPS X3, X1").
		Raw("ADDQ $32, SI").
		Raw("DECQ CX").
		Raw("JNZ b8loop").
		Raw("MOVUPS X0, (DI)").
		Raw("MOVUPS X1, 16(DI)").
		Ret()
	return b.Func()
}

// sqrtKernel builds sqrtSSE2(dst, src *float64, n int).
func sqrtKernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "src", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc("sqrtSSE2", sig, 0)
	b.LoadArg("dst", "DI").
		LoadArg("src", "SI").
		LoadArg("n", "CX").
		// Main loop: 8 doubles (64 bytes) per iteration, packed SQRTPD.
		Raw("sblock:").
		Raw("CMPQ CX, $8").
		Raw("JL stail").
		Raw("MOVUPD (SI), X0").
		Raw("MOVUPD 16(SI), X1").
		Raw("MOVUPD 32(SI), X2").
		Raw("MOVUPD 48(SI), X3").
		Raw("SQRTPD X0, X0").
		Raw("SQRTPD X1, X1").
		Raw("SQRTPD X2, X2").
		Raw("SQRTPD X3, X3").
		Raw("MOVUPD X0, (DI)").
		Raw("MOVUPD X1, 16(DI)").
		Raw("MOVUPD X2, 32(DI)").
		Raw("MOVUPD X3, 48(DI)").
		Raw("ADDQ $64, SI").
		Raw("ADDQ $64, DI").
		Raw("SUBQ $8, CX").
		Raw("JMP sblock").
		// Scalar tail: remaining (n mod 8) elements via SQRTSD.
		Raw("stail:").
		Raw("TESTQ CX, CX").
		Raw("JZ sdone").
		Raw("MOVSD (SI), X0").
		Raw("SQRTSD X0, X0").
		Raw("MOVSD X0, (DI)").
		Raw("ADDQ $8, SI").
		Raw("ADDQ $8, DI").
		Raw("DECQ CX").
		Raw("JMP stail").
		Raw("sdone:").
		Ret()
	return b.Func()
}

// extremeKernel builds maxSSE2/minSSE2(a *float64, n int) float64 from the
// packed (vec) and scalar (sca) extreme mnemonics. a is non-empty (caller
// guarantees n>=1). X0..X3 hold the running packed extreme, X8 the OR of every
// CMPPD-unordered ($3) mask (records "saw a NaN"); the fold returns NaN if any
// lane ever held one, otherwise the reduced extreme.
func extremeKernel(name, vec, sca string) *emit.Function {
	b := amd64.NewFunc(name, reduceSig(), 0)
	b.LoadArg("a", "AX")
	b.LoadArg("n", "CX")
	// Seed the four packed accumulators with a[0] broadcast to both lanes.
	b.Raw("MOVDDUP (AX), X0")
	b.Raw("MOVAPS X0, X1")
	b.Raw("MOVAPS X0, X2")
	b.Raw("MOVAPS X0, X3")
	// X8 = NaN-seen mask, start clear; then fold a[0]'s own NaN-ness in.
	b.Raw("XORPS X8, X8")
	nanScan(b, "X0")
	// Main loop: 8 doubles per iteration.
	b.Raw("eblock:")
	b.Raw("CMPQ CX, $8")
	b.Raw("JL efold")
	b.Raw("MOVUPD (AX), X4")
	b.Raw("MOVUPD 16(AX), X5")
	b.Raw("MOVUPD 32(AX), X6")
	b.Raw("MOVUPD 48(AX), X7")
	nanScan(b, "X4")
	nanScan(b, "X5")
	nanScan(b, "X6")
	nanScan(b, "X7")
	b.Raw(vec + " X4, X0")
	b.Raw(vec + " X5, X1")
	b.Raw(vec + " X6, X2")
	b.Raw(vec + " X7, X3")
	b.Raw("ADDQ $64, AX")
	b.Raw("SUBQ $8, CX")
	b.Raw("JMP eblock")
	// Fold the four accumulators, then the two lanes, into X0 low.
	b.Raw("efold:")
	b.Raw(vec + " X1, X0")
	b.Raw(vec + " X3, X2")
	b.Raw(vec + " X2, X0")
	b.Raw("MOVAPS X0, X1")
	b.Raw("UNPCKHPD X1, X1")
	b.Raw(sca + " X1, X0")
	// Scalar tail: remaining (n mod 8) elements.
	b.Raw("etail:")
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ enan")
	b.Raw("MOVSD (AX), X4")
	nanScan(b, "X4")
	b.Raw(sca + " X4, X0")
	b.Raw("ADDQ $8, AX")
	b.Raw("DECQ CX")
	b.Raw("JMP etail")
	// If any lane of X8 is set, some element was NaN: force the result to NaN.
	b.Raw("enan:")
	b.Raw("MOVAPS X8, X9")
	b.Raw("UNPCKHPD X9, X9")
	b.Raw("ORPD X9, X8")
	b.Raw("MOVQ X8, DX")
	b.Raw("TESTQ DX, DX")
	b.Raw("JZ edone")
	b.Raw("XORPS X1, X1")
	b.Raw("DIVSD X1, X1") // 0.0/0.0 = NaN
	b.Raw("MOVSD X1, X0")
	b.Raw("edone:")
	b.StoreRet("X0", "ret")
	b.Ret()
	return b.Func()
}

// binKernel builds an elementwise binary float64 kernel
//
//	NAME(dst, a, b *float64, n int)   dst[i] = a[i] OP b[i]
//
// from the packed (vec) and scalar (sca) SSE2 mnemonics (e.g. ADDPD/ADDSD).
// The loop processes 8 doubles (64 bytes) per iteration across four XMM
// registers — enough to keep the unit busy on this memory-bandwidth-bound work
// without spilling — then a scalar tail handles the (n mod 8) remainder.
//
// These ops are exact, lane-independent elementwise arithmetic: SSE2's packed
// ADD/SUB/MUL/DIV produce the identical IEEE-754 result the scalar oracle does
// for each element, regardless of grouping (there is no reduction here), so the
// kernel is bit-identical to the scalar Add/Sub/Mul/Div loop.
func binKernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "a", "b", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "DI")
	b.LoadArg("a", "SI")
	b.LoadArg("b", "DX")
	b.LoadArg("n", "CX")
	// Main loop: 8 doubles per iteration. Load a[0:8] into X0..X3, OP with
	// b[0:8], store into dst[0:8].
	b.Raw("bblock:")
	b.Raw("CMPQ CX, $8")
	b.Raw("JL btail")
	b.Raw("MOVUPD (SI), X0")
	b.Raw("MOVUPD 16(SI), X1")
	b.Raw("MOVUPD 32(SI), X2")
	b.Raw("MOVUPD 48(SI), X3")
	b.Raw("MOVUPD (DX), X4")
	b.Raw("MOVUPD 16(DX), X5")
	b.Raw("MOVUPD 32(DX), X6")
	b.Raw("MOVUPD 48(DX), X7")
	b.Raw(vec + " X4, X0")
	b.Raw(vec + " X5, X1")
	b.Raw(vec + " X6, X2")
	b.Raw(vec + " X7, X3")
	b.Raw("MOVUPD X0, (DI)")
	b.Raw("MOVUPD X1, 16(DI)")
	b.Raw("MOVUPD X2, 32(DI)")
	b.Raw("MOVUPD X3, 48(DI)")
	b.Raw("ADDQ $64, SI")
	b.Raw("ADDQ $64, DX")
	b.Raw("ADDQ $64, DI")
	b.Raw("SUBQ $8, CX")
	b.Raw("JMP bblock")
	// Scalar tail: remaining (n mod 8) elements via the scalar mnemonic.
	b.Raw("btail:")
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ bdone")
	b.Raw("MOVSD (SI), X0")
	b.Raw("MOVSD (DX), X1")
	b.Raw(sca + " X1, X0")
	b.Raw("MOVSD X0, (DI)")
	b.Raw("ADDQ $8, SI")
	b.Raw("ADDQ $8, DX")
	b.Raw("ADDQ $8, DI")
	b.Raw("DECQ CX")
	b.Raw("JMP btail")
	b.Raw("bdone:")
	b.Ret()
	return b.Func()
}

// sqrt32Kernel builds sqrt32SSE2(dst, src *float32, n int): SQRTPS over 16
// floats per iteration, SQRTSS for the tail. Both are the correctly rounded
// IEEE float32 square root, which is float32(math.Sqrt(float64(x))) bit for
// bit (float64 carries more than twice float32's precision).
func sqrt32Kernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "src", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc("sqrt32SSE2", sig, 0)
	b.LoadArg("dst", "DI").LoadArg("src", "SI").LoadArg("n", "CX")
	b.Raw("qblock:").Raw("CMPQ CX, $16").Raw("JL qtail")
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS %d(SI), X%d", 16*r, r)
	}
	for r := 0; r < 4; r++ {
		b.Raw("SQRTPS X%d, X%d", r, r)
	}
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS X%d, %d(DI)", r, 16*r)
	}
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DI").Raw("SUBQ $16, CX").Raw("JMP qblock").
		Raw("qtail:").Raw("TESTQ CX, CX").Raw("JZ qdone").
		Raw("MOVSS (SI), X0").Raw("SQRTSS X0, X0").Raw("MOVSS X0, (DI)").
		Raw("ADDQ $4, SI").Raw("ADDQ $4, DI").Raw("DECQ CX").Raw("JMP qtail").
		Raw("qdone:")
	b.Ret()
	return b.Func()
}

// bin32Kernel builds NAME(dst, a, b *float32, n int): dst[i] = a[i] OP b[i],
// 16 floats per iteration over four XMM registers with the packed op, then a
// scalar tail with sca. Each lane is one IEEE float32 operation, so the result
// is the scalar loop's bit for bit.
func bin32Kernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout(
		[]string{"dst", "a", "b", "n"},
		[]amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "DI").LoadArg("a", "SI").LoadArg("b", "DX").LoadArg("n", "CX")
	b.Raw("fblock:").Raw("CMPQ CX, $16").Raw("JL ftail")
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS %d(SI), X%d", 16*r, r)
	}
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS %d(DX), X%d", 16*r, 4+r)
	}
	for r := 0; r < 4; r++ {
		b.Raw("%s X%d, X%d", vec, 4+r, r) // X(r) = a OP b
	}
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS X%d, %d(DI)", r, 16*r)
	}
	b.Raw("ADDQ $64, SI").Raw("ADDQ $64, DX").Raw("ADDQ $64, DI").Raw("SUBQ $16, CX").Raw("JMP fblock").
		Raw("ftail:").Raw("TESTQ CX, CX").Raw("JZ fdone").
		Raw("MOVSS (SI), X0").Raw("MOVSS (DX), X1").
		Raw("%s X1, X0", sca).
		Raw("MOVSS X0, (DI)").
		Raw("ADDQ $4, SI").Raw("ADDQ $4, DX").Raw("ADDQ $4, DI").Raw("DECQ CX").Raw("JMP ftail").
		Raw("fdone:")
	b.Ret()
	return b.Func()
}

// gemmMR, gemmNR are the micro-tile the packing in gemm.go produces on amd64:
// 6 rows of packed A per k step, 8 columns of packed B.
const (
	gemmMR = 6
	gemmNR = 8
)

func gemmSig() amd64.Signature {
	return amd64.Layout(
		[]string{"kc", "pa", "pb", "c", "ldc"},
		[]amd64.Type{amd64.Int64, amd64.Ptr, amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
}

// gemmFMAKernel builds gemmMicro6x8FMA(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*6+r] * pb[p*8+col]   (r<6, col<8)
//
// the BLIS Haswell dgemm shape (6x8, row-major here). Each k step loads the
// 8-double B row into two YMM, broadcasts each of the 6 A values with
// VBROADCASTSD straight from the panel, and issues 12 VFMADD231PD into the 12
// accumulators Y0..Y11 (row r in Y(2r), Y(2r+1)). Twelve independent chains
// cover the FMA latency on both ports (Haswell: 5 cycles x 2 ports needs >= 10),
// and 8 loads per 12 FMAs stay under the 2-loads-per-cycle budget, so the loop
// is FMA-throughput bound — the property that makes it a real dgemm kernel.
//
// Fused: each step rounds a*b+c once, as tuned BLAS (and gc's own scalar code at
// GOAMD64=v3, and arm64's) do. The SSE2 fallback below does not fuse, so on
// non-integer data the two can differ in the last place; both are the ikj sum.
// The caller gates this kernel on hasFMA (CPUID FMA+AVX+OSXSAVE and XCR0 YMM).
func gemmFMAKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x8FMA", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI") // A panel
	b.LoadArg("pb", "DI") // B panel
	b.LoadArg("c", "DX")  // C base
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $3, R8") // ldc -> bytes
	// Prefetch the C tile now: its rows are ldc apart, so they are 6 different
	// cache lines (12 when a row straddles two), usually not in L1, and the
	// store at the end would otherwise stall on each of them in turn. Issued
	// first, they arrive while the k loop runs. The last byte of each row is
	// fetched too, for a row that does not start on a line boundary.
	b.Raw("MOVQ DX, R9")
	for r := 0; r < gemmMR; r++ {
		b.Raw("PREFETCHT0 (R9)")
		b.Raw("PREFETCHT0 63(R9)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, R9")
		}
	}
	for r := 0; r < 12; r++ {
		b.Raw(fmt.Sprintf("VXORPD Y%d, Y%d, Y%d", r, r, r))
	}
	// Unrolled by two k steps, with a one-step tail. The A panel streams from
	// L2 (the B micro-panel stays in L1 across the whole A block), so each pair
	// of steps prefetches the A line 8 steps ahead.
	b.Raw("MOVQ CX, R10")
	b.Raw("SHRQ $1, R10")
	b.Raw("JZ ftail")
	b.Raw("fkloop:")
	b.Raw(fmt.Sprintf("PREFETCHT0 %d(SI)", 8*gemmMR*8))
	fmaStep(b, 0)
	fmaStep(b, 1)
	b.Raw(fmt.Sprintf("ADDQ $%d, SI", 2*8*gemmMR))
	b.Raw(fmt.Sprintf("ADDQ $%d, DI", 2*8*gemmNR))
	b.Raw("DECQ R10")
	b.Raw("JNZ fkloop")
	b.Raw("ftail:")
	b.Raw("TESTQ $1, CX")
	b.Raw("JZ fstore")
	fmaStep(b, 0)
	b.Raw("fstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw(fmt.Sprintf("VADDPD (DX), Y%d, Y15", 2*r))
		b.Raw("VMOVUPD Y15, (DX)")
		b.Raw(fmt.Sprintf("VADDPD 32(DX), Y%d, Y15", 2*r+1))
		b.Raw("VMOVUPD Y15, 32(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Raw("VZEROUPPER") // no AVX-SSE transition penalty in the Go code after
	b.Ret()
	return b.Func()
}

// gemm32NR is the float32 tile width on amd64: two YMM of eight floats.
const gemm32NR = 16

// gemm32FMAKernel builds gemmMicro6x16FMA(kc int, pa, pb, c *float32, ldc int),
// the float32 micro-kernel of the packed GEMM:
//
//	for p in [0,kc):  C[r][col] += pa[p*6+r] * pb[p*16+col]   (r<6, col<16)
//
// gemmFMAKernel at twice the lanes, the BLIS Haswell sgemm shape: the 16-float
// B row in Y12,Y13, each of the six A values broadcast with VBROADCASTSS, and
// twelve VFMADD231PS into Y0..Y11 (row r in Y(2r), Y(2r+1)). Gated on hasFMA.
func gemm32FMAKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x16FMA", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI")
	b.LoadArg("pb", "DI")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $2, R8") // ldc -> bytes
	// Prefetch the C tile: six rows of 64 bytes, each on one or two lines.
	b.Raw("MOVQ DX, R9")
	for r := 0; r < gemmMR; r++ {
		b.Raw("PREFETCHT0 (R9)")
		b.Raw("PREFETCHT0 63(R9)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, R9")
		}
	}
	for r := 0; r < 12; r++ {
		b.Raw(fmt.Sprintf("VXORPS Y%d, Y%d, Y%d", r, r, r))
	}
	b.Raw("MOVQ CX, R10")
	b.Raw("SHRQ $1, R10")
	b.Raw("JZ stail")
	b.Raw("skloop:")
	b.Raw(fmt.Sprintf("PREFETCHT0 %d(SI)", 16*gemmMR*4))
	fma32Step(b, 0)
	fma32Step(b, 1)
	b.Raw(fmt.Sprintf("ADDQ $%d, SI", 2*4*gemmMR))
	b.Raw(fmt.Sprintf("ADDQ $%d, DI", 2*4*gemm32NR))
	b.Raw("DECQ R10")
	b.Raw("JNZ skloop")
	b.Raw("stail:")
	b.Raw("TESTQ $1, CX")
	b.Raw("JZ sstore")
	fma32Step(b, 0)
	b.Raw("sstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw(fmt.Sprintf("VADDPS (DX), Y%d, Y15", 2*r))
		b.Raw("VMOVUPS Y15, (DX)")
		b.Raw(fmt.Sprintf("VADDPS 32(DX), Y%d, Y15", 2*r+1))
		b.Raw("VMOVUPS Y15, 32(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Raw("VZEROUPPER")
	b.Ret()
	return b.Func()
}

// gemm32UAKernel builds gemmUA<mr>x16FMA(kc int, a *float32, lda int, pb, c
// *float32, ldc int): the 6x16 float32 tile with A read in place instead of
// packed, for mr rows (1..6):
//
//	for p in [0,kc):  C[r][col] += a[r*lda+p] * pb[p*16+col]   (r<mr, col<16)
//
// It is the "rv" sup kernel of BLIS (bli_gemmsup_rv_haswell_asm_s6x16m.c):
// each A value is broadcast straight from its row, r*lda bytes past the
// k-step's column, through a scaled index (lda, 2lda, 3lda in R11, 4lda,
// 5lda in R12), and the column advances 4 bytes per step. A small product
// then pays no copy of A at all. B stays packed, so a partial last panel is
// still zero-padded; the kernels for mr < 6 cover the bottom rows of a block
// without reading past A's last row.
//
// gemmUAKernel(mr, false) is the float64 counterpart over the 6x8 packing,
// gemmUA<mr>x8FMA: the same loop in PD instructions, 8-byte steps.
func gemm32UAKernel(mr int) *emit.Function { return gemmUAKernel(mr, true) }

func gemmUAKernel(mr int, single bool) *emit.Function {
	sig := amd64.Layout(
		[]string{"kc", "a", "lda", "pb", "c", "ldc"},
		[]amd64.Type{amd64.Int64, amd64.Ptr, amd64.Int64, amd64.Ptr, amd64.Ptr, amd64.Int64},
		nil, nil,
	)
	name, size, shift, sfx, bcast := fmt.Sprintf("gemmUA%dx8FMA", mr), 8, 3, "PD", "VBROADCASTSD"
	if single {
		name, size, shift, sfx, bcast = fmt.Sprintf("gemmUA%dx16FMA", mr), 4, 2, "PS", "VBROADCASTSS"
	}
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("a", "SI")
	b.LoadArg("lda", "R8")
	b.LoadArg("pb", "DI")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R9")
	b.Raw(fmt.Sprintf("SHLQ $%d, R8", shift)) // lda -> bytes
	b.Raw(fmt.Sprintf("SHLQ $%d, R9", shift)) // ldc -> bytes
	b.Raw("LEAQ (R8)(R8*2), R11")             // 3*lda
	b.Raw("LEAQ (R8)(R8*4), R12")             // 5*lda
	b.Raw("MOVQ DX, R10")
	for r := 0; r < mr; r++ {
		b.Raw("PREFETCHT0 (R10)")
		b.Raw("PREFETCHT0 63(R10)")
		if r < mr-1 {
			b.Raw("ADDQ R9, R10")
		}
	}
	for r := 0; r < 2*mr; r++ {
		b.Raw(fmt.Sprintf("VXOR%s Y%d, Y%d, Y%d", sfx, r, r, r))
	}
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ uastore")
	b.Raw("uakloop:")
	b.Raw(fmt.Sprintf("VMOVU%s (DI), Y12", sfx))
	b.Raw(fmt.Sprintf("VMOVU%s 32(DI), Y13", sfx))
	for r := 0; r < mr; r++ {
		b.Raw(fmt.Sprintf("%s %s, Y14", bcast, uaRow(r)))
		b.Raw(fmt.Sprintf("VFMADD231%s Y12, Y14, Y%d", sfx, 2*r))
		b.Raw(fmt.Sprintf("VFMADD231%s Y13, Y14, Y%d", sfx, 2*r+1))
	}
	b.Raw(fmt.Sprintf("ADDQ $%d, SI", size))
	b.Raw("ADDQ $64, DI") // one packed B row: 16 floats or 8 doubles
	b.Raw("DECQ CX")
	b.Raw("JNZ uakloop")
	b.Raw("uastore:")
	for r := 0; r < mr; r++ {
		b.Raw(fmt.Sprintf("VADD%s (DX), Y%d, Y15", sfx, 2*r))
		b.Raw(fmt.Sprintf("VMOVU%s Y15, (DX)", sfx))
		b.Raw(fmt.Sprintf("VADD%s 32(DX), Y%d, Y15", sfx, 2*r+1))
		b.Raw(fmt.Sprintf("VMOVU%s Y15, 32(DX)", sfx))
		if r < mr-1 {
			b.Raw("ADDQ R9, DX")
		}
	}
	b.Raw("VZEROUPPER")
	b.Ret()
	return b.Func()
}

// uaRow is the address of row r of the unpacked A column at SI, rows lda
// bytes apart (R8), 3*lda in R11 and 5*lda in R12.
func uaRow(r int) string {
	return [...]string{"(SI)", "(SI)(R8*1)", "(SI)(R8*2)", "(SI)(R11*1)", "(SI)(R8*4)", "(SI)(R12*1)"}[r]
}

// fma32Step emits one k step of the 6x16 float32 FMA tile.
func fma32Step(b *amd64.Builder, step int) {
	ao, bo := 4*gemmMR*step, 4*gemm32NR*step
	b.Raw(fmt.Sprintf("VMOVUPS %d(DI), Y12", bo))    // B cols 0..7
	b.Raw(fmt.Sprintf("VMOVUPS %d(DI), Y13", bo+32)) // B cols 8..15
	for r := 0; r < gemmMR; r++ {
		b.Raw(fmt.Sprintf("VBROADCASTSS %d(SI), Y14", ao+4*r))
		b.Raw(fmt.Sprintf("VFMADD231PS Y12, Y14, Y%d", 2*r))
		b.Raw(fmt.Sprintf("VFMADD231PS Y13, Y14, Y%d", 2*r+1))
	}
}

// dotFMAKernel builds dotFMA(a, b *float64, n int) float64, the inner product
// with four YMM accumulators (16 doubles per iteration, one VFMADD231PD per 4
// doubles reading b straight from memory): two loads per FMA make the loop
// load-bound, and four chains cover the FMA latency at that rate. The
// accumulators are folded pairwise, then the four lanes, then the tail is
// fused in one element at a time. Gated on hasFMA like the GEMM kernel.
func dotFMAKernel() *emit.Function {
	sig := amd64.Layout(
		[]string{"a", "b", "n"}, []amd64.Type{amd64.Ptr, amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float64},
	)
	b := amd64.NewFunc("dotFMA", sig, 0)
	b.LoadArg("a", "SI").LoadArg("b", "DI").LoadArg("n", "CX")
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("VXORPD Y%d, Y%d, Y%d", r, r, r))
	}
	b.Raw("dloop:")
	b.Raw("CMPQ CX, $16")
	b.Raw("JL dfold")
	for r := 0; r < 4; r++ {
		b.Raw(fmt.Sprintf("VMOVUPD %d(SI), Y%d", 32*r, 4+r))
		b.Raw(fmt.Sprintf("VFMADD231PD %d(DI), Y%d, Y%d", 32*r, 4+r, r))
	}
	b.Raw("ADDQ $128, SI")
	b.Raw("ADDQ $128, DI")
	b.Raw("SUBQ $16, CX")
	b.Raw("JMP dloop")
	b.Raw("dfold:")
	b.Raw("VADDPD Y1, Y0, Y0")
	b.Raw("VADDPD Y3, Y2, Y2")
	b.Raw("VADDPD Y2, Y0, Y0")
	b.Raw("VEXTRACTF128 $1, Y0, X1")
	b.Raw("VADDPD X1, X0, X0")
	b.Raw("VUNPCKHPD X0, X0, X1")
	b.Raw("VADDSD X1, X0, X0")
	b.Raw("dtail:")
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ ddone")
	b.Raw("VMOVSD (SI), X1")
	b.Raw("VFMADD231SD (DI), X1, X0")
	b.Raw("ADDQ $8, SI")
	b.Raw("ADDQ $8, DI")
	b.Raw("DECQ CX")
	b.Raw("JMP dtail")
	b.Raw("ddone:")
	b.Raw("VZEROUPPER")
	b.StoreRet("X0", "ret")
	b.Ret()
	return b.Func()
}

// fmaStep emits one k step of the 6x8 FMA tile, step reading the A and B
// values at that many steps past SI and DI.
func fmaStep(b *amd64.Builder, step int) {
	ao, bo := 8*gemmMR*step, 8*gemmNR*step
	b.Raw(fmt.Sprintf("VMOVUPD %d(DI), Y12", bo))    // B cols 0..3
	b.Raw(fmt.Sprintf("VMOVUPD %d(DI), Y13", bo+32)) // B cols 4..7
	for r := 0; r < gemmMR; r++ {
		b.Raw(fmt.Sprintf("VBROADCASTSD %d(SI), Y14", ao+8*r))
		b.Raw(fmt.Sprintf("VFMADD231PD Y12, Y14, Y%d", 2*r))
		b.Raw(fmt.Sprintf("VFMADD231PD Y13, Y14, Y%d", 2*r+1))
	}
}

// gemmSSE2HalfKernel builds gemmMicro6x4SSE2(kc int, pa, pb, c *float64, ldc
// int), the baseline (GOAMD64=v1, no FMA) fallback over the same 6x8 packing:
// it computes the 6-row x 4-col half of the tile whose B columns start at pb,
// so the Go side calls it twice (pb, c) and (pb+4, c+4). A whole 6x8 tile in
// XMM would need 24 accumulators; the half needs 12 (X0..X11, row r in X(2r),
// X(2r+1)) plus the B half-row X12,X13, the broadcast A X14 and a product X15.
// The B stride is still the full NR=8 doubles per k step.
//
// SSE2 has no fused multiply-add, so each step is MULPD then ADDPD: the
// running sum is the plain dst += a*b in ikj order, like the scalar oracle at
// GOAMD64=v1.
func gemmSSE2HalfKernel() *emit.Function {
	b := amd64.NewFunc("gemmMicro6x4SSE2", gemmSig(), 0)
	b.LoadArg("kc", "CX")
	b.LoadArg("pa", "SI")
	b.LoadArg("pb", "DI")
	b.LoadArg("c", "DX")
	b.LoadArg("ldc", "R8")
	b.Raw("SHLQ $3, R8")
	for r := 0; r < 12; r++ {
		b.Raw(fmt.Sprintf("XORPS X%d, X%d", r, r))
	}
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ hstore")
	b.Raw("hkloop:")
	b.Raw("MOVUPD (DI), X12")
	b.Raw("MOVUPD 16(DI), X13")
	for r := 0; r < gemmMR; r++ {
		b.Raw(fmt.Sprintf("MOVDDUP %d(SI), X14", 8*r)) // X14 = [a,a]
		b.Raw("MOVAPS X12, X15")
		b.Raw("MULPD X14, X15")
		b.Raw(fmt.Sprintf("ADDPD X15, X%d", 2*r))
		b.Raw("MULPD X13, X14") // the broadcast is spent: reuse it for the product
		b.Raw(fmt.Sprintf("ADDPD X14, X%d", 2*r+1))
	}
	b.Raw(fmt.Sprintf("ADDQ $%d, SI", 8*gemmMR))
	b.Raw(fmt.Sprintf("ADDQ $%d, DI", 8*gemmNR))
	b.Raw("DECQ CX")
	b.Raw("JNZ hkloop")
	b.Raw("hstore:")
	for r := 0; r < gemmMR; r++ {
		b.Raw("MOVUPD (DX), X14")
		b.Raw(fmt.Sprintf("ADDPD X%d, X14", 2*r))
		b.Raw("MOVUPD X14, (DX)")
		b.Raw("MOVUPD 16(DX), X15")
		b.Raw(fmt.Sprintf("ADDPD X%d, X15", 2*r+1))
		b.Raw("MOVUPD X15, 16(DX)")
		if r < gemmMR-1 {
			b.Raw("ADDQ R8, DX")
		}
	}
	b.Ret()
	return b.Func()
}

// nanScan emits: X9 = (reg unordered reg) ? all-ones : 0 ; X8 |= X9, i.e. it
// ORs into the NaN-seen mask X8 a lane of all-ones wherever reg holds a NaN.
// extremum32Kernel builds max32SSE2/min32SSE2(a *float32, n int) float32, for
// n >= 1: the extremum over 16 floats per iteration in four XMM accumulators
// (MAXPS/MINPS), with X8 the OR of every CMPPS-unordered mask so a NaN anywhere
// gives NaN. MAXPS/MINPS return their source operand on a zero tie, so a zero
// result may carry the wrong sign; the Go wrapper hands that case to the scalar
// loop, as maxSIMD does.
func extremum32Kernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout([]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float32})
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("a", "AX").LoadArg("n", "CX")
	// Seed the accumulators with a[0] in every lane.
	b.Raw("MOVSS (AX), X0").Raw("SHUFPS $0x00, X0, X0").
		Raw("MOVAPS X0, X1").Raw("MOVAPS X0, X2").Raw("MOVAPS X0, X3").
		Raw("XORPS X8, X8")
	nanScan32(b, "X0")
	b.Raw("xblock:").Raw("CMPQ CX, $16").Raw("JL xfold")
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS %d(AX), X%d", 16*r, 4+r)
	}
	for r := 0; r < 4; r++ {
		nanScan32(b, fmt.Sprintf("X%d", 4+r))
	}
	for r := 0; r < 4; r++ {
		b.Raw("%s X%d, X%d", vec, 4+r, r)
	}
	b.Raw("ADDQ $64, AX").Raw("SUBQ $16, CX").Raw("JMP xblock")
	// Fold the accumulators, then the four lanes, into X0's low lane.
	b.Raw("xfold:").
		Raw("%s X1, X0", vec).Raw("%s X3, X2", vec).Raw("%s X2, X0", vec).
		Raw("MOVAPS X0, X1").Raw("MOVHLPS X1, X1").Raw("%s X1, X0", vec).
		Raw("MOVAPS X0, X1").Raw("SHUFPS $0x55, X1, X1").Raw("%s X1, X0", sca).
		Raw("xtail:").Raw("TESTQ CX, CX").Raw("JZ xnan").
		Raw("MOVSS (AX), X4")
	nanScan32(b, "X4")
	b.Raw("%s X4, X0", sca).
		Raw("ADDQ $4, AX").Raw("DECQ CX").Raw("JMP xtail").
		// Any lane of X8 set: some element was NaN.
		Raw("xnan:").Raw("MOVMSKPS X8, DX").Raw("TESTQ DX, DX").Raw("JZ xdone").
		Raw("XORPS X1, X1").Raw("DIVSS X1, X1"). // 0/0 = NaN
		Raw("MOVSS X1, X0").
		Raw("xdone:").
		StoreRet("X0", "ret").
		Ret()
	return b.Func()
}

// nanScan32 ORs the unordered (NaN) lanes of reg into X8.
func nanScan32(b *amd64.Builder, reg string) {
	b.Raw("MOVAPS %s, X9", reg).Raw("CMPPS X9, X9, $3").Raw("ORPS X9, X8")
}

func nanScan(b *amd64.Builder, reg string) {
	b.Raw("MOVAPS " + reg + ", X9")
	b.Raw("CMPPD X9, X9, $3")
	b.Raw("ORPD X9, X8")
}
