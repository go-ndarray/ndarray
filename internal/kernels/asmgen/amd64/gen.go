//go:build ignore

// Command gen produces sum_amd64.s, the SSE2 float64 sum-reduction kernel, via
// go-asmgen. Run with: go run gen.go (or `go generate` from the kernels
// package).
//
//   sumSSE2(a *float64, n int) float64
//     Sums a[0..n) with FOUR independent XMM accumulators (8 float64 lanes via
//     2 doubles per register), then folds them and adds a scalar tail. SSE2 is
//     part of the amd64 baseline (GOAMD64=v1), so the kernel is always callable
//     with no CPU-feature branch. Like the arm64 kernel, the lane-parallel
//     grouping is a different-but-valid summation order, so it is validated to a
//     tight relative tolerance against the scalar oracle (and numpy), not held
//     bit-identical. ADDPD rounds each packed add exactly as the scalar add
//     does; only the grouping differs.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")

	sig := amd64.Layout(
		[]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float64},
	)
	b := amd64.NewFunc("sumSSE2", sig, 0)
	b.LoadArg("a", "AX").
		LoadArg("n", "CX").
		// Zero the four packed-double accumulators X0..X3.
		Raw("XORPS X0, X0").
		Raw("XORPS X1, X1").
		Raw("XORPS X2, X2").
		Raw("XORPS X3, X3").
		// Main loop: 8 doubles (64 bytes) per iteration.
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
		// Fold the four accumulators into X0, then its two lanes into the low one.
		Raw("fold:").
		Raw("ADDPD X1, X0").
		Raw("ADDPD X3, X2").
		Raw("ADDPD X2, X0").
		Raw("MOVAPS X0, X1").
		Raw("UNPCKHPD X1, X1"). // X1 low = high lane of X0
		Raw("ADDSD X1, X0").    // X0 low = lane0 + lane1
		// Scalar tail: remaining (n mod 8) elements, accumulated into X0 low.
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
	f.Add(b.Func())

	if err := os.WriteFile("sum_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_amd64.s")
}
