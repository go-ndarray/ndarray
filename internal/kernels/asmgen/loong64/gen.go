// Command gen produces sum_loong64.s, the LASX float64 kernels for loong64, via
// go-asmgen. Run with: go run . (or `go generate` from the kernels package).
//
// Go's loong64 assembler has the LASX vector float64 add/sub/mul/div
// (XVADDD, XVSUBD, XVMULD, XVDIVD), the square root (XVFSQRTD) and the
// 256-bit loads and stores (XVMOVQ, with immediate offsets), and its
// broadcast load is the arrangement form XVMOVQ off(R), X.V4 (xvldrepl.d). It
// has no fused multiply-add: that comes from go-asmgen's XVFMADDD, which
// emits a WORD (transitional until cmd/asm names it). LASX is not on every
// LoongArch CPU, so the Go side calls these kernels only when the kernel
// reports it (AT_HWCAP).
//
// Registers: R4.. hold the arguments; R12..R15 are scratch; X0..X31 are the
// 256-bit LASX registers (four float64 lanes), F0..F31 their low lanes.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

const (
	gemmMR = 8
	gemmNR = 8
)

func main() {
	f := emit.NewFile("loong64")
	f.Add(sumKernel())
	f.Add(dotKernel())
	f.Add(sqrtKernel())
	for _, k := range []struct{ name, vec, sca string }{
		{"addLASX", "XVADDD", "ADDD"}, {"subLASX", "XVSUBD", "SUBD"},
		{"mulLASX", "XVMULD", "MULD"}, {"divLASX", "XVDIVD", "DIVD"},
	} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	f.Add(gemmKernel())
	if err := os.WriteFile("sum_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_loong64.s")
}

// fold adds X0..X3 into X0, then its four lanes into F0.
func fold(b *loong64.Builder) {
	b.Raw("XVADDD X1, X0, X0").Raw("XVADDD X3, X2, X2").Raw("XVADDD X2, X0, X0")
	for l := 0; l < 4; l++ {
		b.Raw("XVMOVQ X0.V[%d], R%d", l, 12+l)
	}
	b.Raw("MOVV R12, F0").Raw("MOVV R13, F1").Raw("MOVV R14, F2").Raw("MOVV R15, F3")
	b.Raw("ADDD F1, F0").Raw("ADDD F3, F2").Raw("ADDD F2, F0")
}

func zero(b *loong64.Builder, n int) {
	for r := 0; r < n; r++ {
		b.Raw("XVXORV X%d, X%d, X%d", r, r, r)
	}
}

// sumKernel builds sumLASX(a *float64, n int) float64: four LASX accumulators
// (16 doubles per iteration), folded, plus a scalar tail. A regrouping of the
// sum, held to a tolerance like the other targets' SIMD sums.
func sumKernel() *emit.Function {
	sig := loong64.Layout([]string{"a", "n"}, []loong64.Type{loong64.Ptr, loong64.Int64},
		[]string{"ret"}, []loong64.Type{loong64.Float64})
	b := loong64.NewFunc("sumLASX", sig, 0)
	b.LoadArg("a", "R4").LoadArg("n", "R5")
	zero(b, 4)
	b.Label("sloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, sfold")
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R4), X%d", 32*k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVADDD X%d, X%d, X%d", 4+k, k, k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $-16, R5").Raw("JMP sloop")
	b.Label("sfold")
	fold(b)
	b.Label("stail").Raw("BEQ R5, sdone").
		Raw("MOVD 0(R4), F1").Raw("ADDD F1, F0").
		Raw("ADDV $8, R4").Raw("ADDV $-1, R5").Raw("JMP stail")
	b.Label("sdone").StoreRet("F0", "ret").Ret()
	return b.Func()
}

// dotKernel builds dotLASX(a, b *float64, n int) float64: four fused
// multiply-add accumulators (xvfmadd.d), folded, plus a fused scalar tail.
func dotKernel() *emit.Function {
	sig := loong64.Layout([]string{"a", "b", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64},
		[]string{"ret"}, []loong64.Type{loong64.Float64})
	b := loong64.NewFunc("dotLASX", sig, 0)
	b.LoadArg("a", "R4").LoadArg("b", "R6").LoadArg("n", "R5")
	zero(b, 4)
	b.Label("dloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, dfold")
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R4), X%d", 32*k, 4+k)
		b.Raw("XVMOVQ %d(R6), X%d", 32*k, 8+k)
	}
	for k := 0; k < 4; k++ {
		b.XVFMADDD(k, 4+k, 8+k, k) // X(k) = a*b + X(k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $128, R6").Raw("ADDV $-16, R5").Raw("JMP dloop")
	b.Label("dfold")
	fold(b)
	b.Label("dtail").Raw("BEQ R5, ddone").
		Raw("MOVD 0(R4), F1").Raw("MOVD 0(R6), F2").Raw("FMADDD F0, F2, F1, F0"). // F0 = F1*F2 + F0; Go's order is Fa, Fk, Fj, Fd
		Raw("ADDV $8, R4").Raw("ADDV $8, R6").Raw("ADDV $-1, R5").Raw("JMP dtail")
	b.Label("ddone").StoreRet("F0", "ret").Ret()
	return b.Func()
}

// sqrtKernel builds sqrtLASX(dst, src *float64, n int): XVFSQRTD over 16
// doubles per iteration, SQRTD for the tail. Both are the correctly rounded
// IEEE square root, so the result is bit-identical to math.Sqrt.
func sqrtKernel() *emit.Function {
	sig := loong64.Layout([]string{"dst", "src", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("sqrtLASX", sig, 0)
	b.LoadArg("dst", "R4").LoadArg("src", "R6").LoadArg("n", "R5")
	b.Label("qloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, qtail")
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R6), X%d", 32*k, k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVFSQRTD X%d, X%d", k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ X%d, %d(R4)", 4+k, 32*k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $128, R6").Raw("ADDV $-16, R5").Raw("JMP qloop")
	b.Label("qtail").Raw("BEQ R5, qdone").
		Raw("MOVD 0(R6), F1").Raw("SQRTD F1, F1").Raw("MOVD F1, 0(R4)").
		Raw("ADDV $8, R4").Raw("ADDV $8, R6").Raw("ADDV $-1, R5").Raw("JMP qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// binKernel builds name(dst, a, b *float64, n int): dst[i] = a[i] OP b[i], 16
// doubles per iteration with the LASX op, a scalar tail with sca. Each lane is
// one correctly rounded IEEE operation, bit-identical to the scalar oracle.
func binKernel(name, vec, sca string) *emit.Function {
	sig := loong64.Layout([]string{"dst", "a", "b", "n"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R4").LoadArg("a", "R6").LoadArg("b", "R7").LoadArg("n", "R5")
	l := name[:3]
	b.Label(l+"loop").Raw("MOVV $16, R12").Raw("BLT R5, R12, %stail", l)
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R6), X%d", 32*k, k)
		b.Raw("XVMOVQ %d(R7), X%d", 32*k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("%s X%d, X%d, X%d", vec, 4+k, k, 8+k) // X(8+k) = X(k) OP X(4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ X%d, %d(R4)", 8+k, 32*k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $128, R6").Raw("ADDV $128, R7").Raw("ADDV $-16, R5").Raw("JMP %sloop", l)
	b.Label(l+"tail").Raw("BEQ R5, %sdone", l).
		Raw("MOVD 0(R6), F1").Raw("MOVD 0(R7), F2").Raw("%s F2, F1", sca).Raw("MOVD F1, 0(R4)").
		Raw("ADDV $8, R4").Raw("ADDV $8, R6").Raw("ADDV $8, R7").Raw("ADDV $-1, R5").Raw("JMP %stail", l)
	b.Label(l + "done").Ret()
	return b.Func()
}

// gemmKernel builds gemmMicro8x8LASX(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*8+col]   (r<8, col<8)
//
// 16 accumulators X0..X15 (row r in X(2r), X(2r+1), four columns each), the B
// row in X16, X17, and each A value broadcast (XVMOVQ .V4, xvldrepl.d) into X18..X25,
// then xvfmadd.d: 16 fused multiply-adds per k step against 10 loads. Fused,
// like the other FMA kernels; on integer-valued data it equals the scalar
// ikj oracle exactly.
func gemmKernel() *emit.Function {
	sig := loong64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]loong64.Type{loong64.Int64, loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("gemmMicro8x8LASX", sig, 0)
	b.LoadArg("kc", "R5").LoadArg("pa", "R6").LoadArg("pb", "R7").LoadArg("c", "R8").LoadArg("ldc", "R9")
	b.Raw("SLLV $3, R9") // ldc -> bytes
	zero(b, 16)
	b.Raw("BEQ R5, gstore")
	b.Label("gloop")
	b.Raw("XVMOVQ 0(R7), X16").Raw("XVMOVQ 32(R7), X17")
	for r := 0; r < gemmMR; r++ {
		b.Raw("XVMOVQ %d(R6), X%d.V4", 8*r, 18+r) // xvldrepl.d: pa[r] in all four lanes
	}
	for r := 0; r < gemmMR; r++ {
		b.XVFMADDD(2*r, 18+r, 16, 2*r)
		b.XVFMADDD(2*r+1, 18+r, 17, 2*r+1)
	}
	b.Raw("ADDV $%d, R6", 8*gemmMR).Raw("ADDV $%d, R7", 8*gemmNR).Raw("ADDV $-1, R5").Raw("BNE R5, gloop")
	b.Label("gstore")
	for r := 0; r < gemmMR; r++ {
		b.Raw("XVMOVQ 0(R8), X16").Raw("XVMOVQ 32(R8), X17")
		b.Raw("XVADDD X%d, X16, X16", 2*r).Raw("XVADDD X%d, X17, X17", 2*r+1)
		b.Raw("XVMOVQ X16, 0(R8)").Raw("XVMOVQ X17, 32(R8)")
		if r < gemmMR-1 {
			b.Raw("ADDV R9, R8")
		}
	}
	b.Ret()
	return b.Func()
}
