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
		// "XVADDD" -> "VADDF", "ADDD" -> "ADDF": the LSX single-precision op.
		f.Add(bin32Kernel(k.name[:3]+"32LSX", k.vec[1:len(k.vec)-1]+"F", k.sca[:len(k.sca)-1]+"F"))
	}
	f.Add(gemmKernel())
	f.Add(gemm32Kernel())
	f.Add(blocks8Kernel())
	f.Add(sqrt32Kernel())
	abs64 := f.Data("abs64", lanes(8, 0x7fffffffffffffff))
	inf64 := f.Data("inf64", lanes(8, 0x7ff0000000000000))
	abs32 := f.Data("abs32", lanes(4, 0x7fffffff))
	inf32 := f.Data("inf32", lanes(4, 0x7f800000))
	f.Add(extremumKernel("maxLSX", 64, true, abs64, inf64))
	f.Add(extremumKernel("minLSX", 64, false, abs64, inf64))
	f.Add(extremumKernel("max32LSX", 32, true, abs32, inf32))
	f.Add(extremumKernel("min32LSX", 32, false, abs32, inf32))
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

// lanes is a 16-byte vector constant: v in every lane of size bytes, each lane
// little-endian.
func lanes(size int, v uint64) []byte {
	b := make([]byte, 0, 16)
	for len(b) < 16 {
		for i := 0; i < size; i++ {
			b = append(b, byte(v>>(8*i)))
		}
	}
	return b
}

// extremumKernel builds NAME(a *floatN, n int, out *[16]byte) int64 for n a
// positive multiple of the block (8 doubles or 16 floats), entirely in LSX
// integer arithmetic: Go's loong64 assembler has no vector float max, min or
// compare.
//
//   - Order: t = x XOR ((x >>a N-1) >>l 1) flips the magnitude bits of the
//     negatives, so the signed order of t is the float order, with -0 below
//     +0, Go's rule. The map is its own inverse.
//   - Extremum: the max of t and acc is acc XOR ((acc XOR t) AND (acc < t))
//     (vslt.d/vslt.w), and the min swaps the compare.
//   - NaN: a lane is NaN iff its magnitude bits (x AND abs) exceed +Inf's
//     bits as an unsigned integer (vsltu). The masks are OR-accumulated, and
//     a non-zero result is returned so the wrapper can answer NaN.
//
// Four accumulators (V0-V3) are folded the same way into one vector, which is
// stored, still in the t order, at out. The Go wrapper maps it back and folds
// the lanes and the tail with Go's max/min. Registers: V4-V7 the block,
// V8-V19 scratch, V20 abs, V21 +Inf, V22 the NaN mask.
func extremumKernel(name string, bits int, isMax bool, abs, inf string) *emit.Function {
	sig := loong64.Layout([]string{"a", "n", "out"},
		[]loong64.Type{loong64.Ptr, loong64.Int64, loong64.Ptr}, []string{"ret"}, []loong64.Type{loong64.Int64})
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R4").LoadArg("n", "R5").LoadArg("out", "R6")
	w, block := "V", 8
	if bits == 32 {
		w, block = "W", 16
	}
	sra, srl, slt, sltu := "VSRA"+w, "VSRL"+w, "VSLT"+w, "VSLT"+w+"U"
	b.Raw("MOVV $%s+0(SB), R12", abs).Raw("VMOVQ (R12), V20").
		Raw("MOVV $%s+0(SB), R12", inf).Raw("VMOVQ (R12), V21").
		Raw("VXORV V22, V22, V22")
	// sel sets acc to the extremum of acc and t, with m and d as scratch.
	sel := func(acc, t, m, d int) {
		if isMax {
			b.Raw("%s V%d, V%d, V%d", slt, t, acc, m) // m = acc < t
		} else {
			b.Raw("%s V%d, V%d, V%d", slt, acc, t, m) // m = t < acc
		}
		b.Raw("VXORV V%d, V%d, V%d", t, acc, d).
			Raw("VANDV V%d, V%d, V%d", m, d, d).
			Raw("VXORV V%d, V%d, V%d", d, acc, acc)
	}
	// load reads one block into V4-V7, ORs its NaN lanes into V22 and leaves
	// the ordered integers in V8-V11.
	load := func() {
		for k := 0; k < 4; k++ {
			b.Raw("VMOVQ %d(R4), V%d", 16*k, 4+k)
		}
		for k := 0; k < 4; k++ {
			b.Raw("VANDV V20, V%d, V%d", 4+k, 12+k).
				Raw("%s V%d, V21, V%d", sltu, 12+k, 12+k). // +Inf < |x|: NaN
				Raw("VORV V%d, V22, V22", 12+k).
				Raw("%s $%d, V%d, V%d", sra, bits-1, 4+k, 8+k).
				Raw("%s $1, V%d, V%d", srl, 8+k, 8+k).
				Raw("VXORV V%d, V%d, V%d", 4+k, 8+k, 8+k)
		}
		b.Raw("ADDV $64, R4").Raw("ADDV $-%d, R5", block)
	}
	load()
	for k := 0; k < 4; k++ {
		b.Raw("VORV V%d, V%d, V%d", 8+k, 8+k, k) // the first block seeds acc
	}
	b.Label(name+"loop").Raw("BEQ R5, %sfold", name)
	load()
	for k := 0; k < 4; k++ {
		sel(k, 8+k, 12+k, 16+k)
	}
	b.Raw("JMP %sloop", name)
	b.Label(name + "fold")
	sel(0, 1, 12, 16)
	sel(2, 3, 13, 17)
	sel(0, 2, 14, 18)
	b.Raw("VMOVQ V0, (R6)").
		Raw("VMOVQ V22.V[0], R13").Raw("VMOVQ V22.V[1], R14").Raw("OR R14, R13").
		StoreRet("R13", "ret").Ret()
	return b.Func()
}

// sqrt32Kernel builds sqrt32LSX(dst, src *float32, n int): vfsqrt.s over four
// 128-bit registers, 16 floats per iteration, SQRTF for the tail; the
// correctly rounded float32 square root, bit-identical to
// float32(math.Sqrt(float64(x))).
func sqrt32Kernel() *emit.Function {
	sig := loong64.Layout([]string{"dst", "src", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("sqrt32LSX", sig, 0)
	b.LoadArg("dst", "R4").LoadArg("src", "R6").LoadArg("n", "R5")
	b.Label("qloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, qtail")
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ %d(R6), V%d", 16*k, k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VFSQRTF V%d, V%d", k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ V%d, %d(R4)", 4+k, 16*k)
	}
	b.Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $-16, R5").Raw("JMP qloop")
	b.Label("qtail").Raw("BEQ R5, qdone").
		Raw("MOVF 0(R6), F1").Raw("SQRTF F1, F1").Raw("MOVF F1, 0(R4)").
		Raw("ADDV $4, R4").Raw("ADDV $4, R6").Raw("ADDV $-1, R5").Raw("JMP qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// bin32Kernel builds name(dst, a, b *float32, n int): dst[i] = a[i] OP b[i],
// 16 floats per iteration over four 128-bit LSX registers (four independent
// chains, as the block kernel's lesson on one LASX register taught), a scalar
// tail with sca. Each lane is one IEEE float32 operation, bit-identical to the
// scalar oracle.
func bin32Kernel(name, vec, sca string) *emit.Function {
	sig := loong64.Layout([]string{"dst", "a", "b", "n"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R4").LoadArg("a", "R6").LoadArg("b", "R7").LoadArg("n", "R5")
	b.Label("floop").Raw("MOVV $16, R12").Raw("BLT R5, R12, ftail")
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ %d(R6), V%d", 16*k, k)
		b.Raw("VMOVQ %d(R7), V%d", 16*k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("%s V%d, V%d, V%d", vec, 4+k, k, 8+k) // V(8+k) = V(k) OP V(4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("VMOVQ V%d, %d(R4)", 8+k, 16*k)
	}
	b.Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $64, R7").Raw("ADDV $-16, R5").Raw("JMP floop")
	b.Label("ftail").Raw("BEQ R5, fdone").
		Raw("MOVF 0(R6), F1").Raw("MOVF 0(R7), F2").Raw("%s F2, F1", sca).Raw("MOVF F1, 0(R4)").
		Raw("ADDV $4, R4").Raw("ADDV $4, R6").Raw("ADDV $4, R7").Raw("ADDV $-1, R5").Raw("JMP ftail")
	b.Label("fdone").Ret()
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

// gemm32Kernel builds gemmMicro8x16LASX(kc int, pa, pb, c *float32, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*16+col]   (r<8, col<16)
//
// gemmKernel at twice the lanes: 16 accumulators X0..X15 (row r in X(2r),
// X(2r+1), eight columns each), the B row in X16, X17, each A value broadcast
// with xvldrepl.w (XVMOVQ .W8) into X18..X25, and xvfmadd.s, which go-asmgen
// encodes (XVFMADDF, v0.23.0) because Go's assembler has no vector float FMA.
func gemm32Kernel() *emit.Function {
	sig := loong64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]loong64.Type{loong64.Int64, loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("gemmMicro8x16LASX", sig, 0)
	b.LoadArg("kc", "R5").LoadArg("pa", "R6").LoadArg("pb", "R7").LoadArg("c", "R8").LoadArg("ldc", "R9")
	b.Raw("SLLV $2, R9") // ldc -> bytes
	zero(b, 16)
	b.Raw("BEQ R5, sstore")
	b.Label("sloop")
	b.Raw("XVMOVQ 0(R7), X16").Raw("XVMOVQ 32(R7), X17")
	for r := 0; r < 8; r++ {
		b.Raw("XVMOVQ %d(R6), X%d.W8", 4*r, 18+r) // xvldrepl.w: pa[r] in all eight lanes
	}
	for r := 0; r < 8; r++ {
		b.XVFMADDF(2*r, 18+r, 16, 2*r)
		b.XVFMADDF(2*r+1, 18+r, 17, 2*r+1)
	}
	b.Raw("ADDV $32, R6").Raw("ADDV $64, R7").Raw("ADDV $-1, R5").Raw("BNE R5, sloop")
	b.Label("sstore")
	for r := 0; r < 8; r++ {
		b.Raw("XVMOVQ 0(R8), X16").Raw("XVMOVQ 32(R8), X17")
		b.Raw("XVADDF X%d, X16, X16", 2*r).Raw("XVADDF X%d, X17, X17", 2*r+1)
		b.Raw("XVMOVQ X16, 0(R8)").Raw("XVMOVQ X17, 32(R8)")
		if r < 7 {
			b.Raw("ADDV R9, R8")
		}
	}
	b.Ret()
	return b.Func()
}

// blocks8Kernel builds blocks8LSX(r, a *float32, blocks int): for each of
// the blocks of eight float32 at a, in order, r[j] += a[8i+j] (NumPy's
// pairwise-sum block). The eight accumulators are split over two 128-bit LSX
// registers, V0 and V1, and VADDF (vfadd.s) is one IEEE float32 addition per
// lane, so the result is the scalar loop's bit for bit. blocks >= 1.
//
// Not one 256-bit LASX register: that version kept every block's addition
// waiting on the previous one, a single dependency chain, and on a Loongson
// 3C5000L it was 26-47% SLOWER than the scalar loop's eight chains. Two
// registers, two chains (as SSE2 on amd64): 24% faster.
func blocks8Kernel() *emit.Function {
	sig := loong64.Layout([]string{"r", "a", "blocks"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("blocks8LSX", sig, 0)
	b.LoadArg("r", "R4").LoadArg("a", "R5").LoadArg("blocks", "R6")
	b.Raw("VMOVQ 0(R4), V0").Raw("VMOVQ 16(R4), V1")
	b.Label("b8loop").
		Raw("VMOVQ 0(R5), V2").
		Raw("VMOVQ 16(R5), V3").
		Raw("VADDF V2, V0, V0").
		Raw("VADDF V3, V1, V1").
		Raw("ADDV $32, R5").
		Raw("ADDV $-1, R6").
		Raw("BNE R6, b8loop")
	b.Raw("VMOVQ V0, 0(R4)").Raw("VMOVQ V1, 16(R4)")
	b.Ret()
	return b.Func()
}
