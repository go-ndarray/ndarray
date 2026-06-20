//go:build ignore

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

// nanScan emits: X9 = (reg unordered reg) ? all-ones : 0 ; X8 |= X9, i.e. it
// ORs into the NaN-seen mask X8 a lane of all-ones wherever reg holds a NaN.
func nanScan(b *amd64.Builder, reg string) {
	b.Raw("MOVAPS " + reg + ", X9")
	b.Raw("CMPPD X9, X9, $3")
	b.Raw("ORPD X9, X8")
}
