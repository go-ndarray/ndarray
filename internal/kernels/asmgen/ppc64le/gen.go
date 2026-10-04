// Command gen produces sum_ppc64le.s, the VSX float64 kernels for ppc64le, via
// go-asmgen. Run with: go run . (or `go generate` from the kernels package).
//
// Go's ppc64 assembler has the VSX loads, stores and splats (LXVD2X, LXVDSX,
// STXVD2X, XXPERMDI) but no vector double arithmetic; the arithmetic comes from
// go-asmgen's VSX encoders (Builder.XVADDDP, XVMADDADP, ...), which emit WORDs.
//
// Lane order: LXVD2X/STXVD2X swap the two doublewords of a vector on
// little-endian. Every vector of one kernel is loaded and stored that way, so a
// lane-wise operation pairs the same elements in every operand and the store
// puts each result back at its own index. LXVDSX (load and splat) puts the same
// double in both lanes, so it needs no care.
//
// Registers: R3.. hold the arguments; R14..R20 hold the byte offsets that
// LXVD2X's indexed form needs (R0 reads as 0); R30/R31/R13/R2/R1 are Go's and
// untouched. VS0..VS31 are accumulators, VS32..VS43 operands.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

const (
	gemmMR = 8
	gemmNR = 8
)

func main() {
	f := emit.NewFile("ppc64le")
	f.Add(sumKernel())
	f.Add(dotKernel())
	f.Add(sqrtKernel())
	for _, k := range []struct {
		name, vec, sca string
	}{{"addVSX", "add", "FADD"}, {"subVSX", "sub", "FSUB"}, {"mulVSX", "mul", "FMUL"}, {"divVSX", "div", "FDIV"}} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	f.Add(gemmKernel())
	if err := os.WriteFile("sum_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_ppc64le.s")
}

// offsets loads the byte offsets 16, 32, 48 into R14, R15, R16 (and, with
// splat, 8..56 into R17..R20 for the A-panel splats): LXVD2X/LXVDSX take only
// a (base)(index) address.
func offsets(b *ppc64.Builder, splat bool) {
	b.Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16")
	if splat {
		b.Raw("MOVD $8, R17").Raw("MOVD $24, R18").Raw("MOVD $40, R19").Raw("MOVD $56, R20")
	}
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

// fold adds VS0..VS3 into VS0 and then its two lanes together, leaving the
// total in both lanes, so F0 (the scalar view of VS0) holds it.
func fold(b *ppc64.Builder) {
	b.XVADDDP(0, 0, 1).XVADDDP(2, 2, 3).XVADDDP(0, 0, 2)
	b.Raw("XXPERMDI VS0, VS0, $2, VS1") // swap the doublewords
	b.XVADDDP(0, 0, 1)
}

func reduceSig() ppc64.Signature {
	return ppc64.Layout([]string{"a", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Int64},
		[]string{"ret"}, []ppc64.Type{ppc64.Float64})
}

// sumKernel builds sumVSX(a *float64, n int) float64: four VSX accumulators
// (8 doubles per iteration), folded, plus a scalar tail. A regrouping of the
// sum, held to a tolerance like the other targets' SIMD sums.
func sumKernel() *emit.Function {
	b := ppc64.NewFunc("sumVSX", reduceSig(), 0)
	b.LoadArg("a", "R3").LoadArg("n", "R4")
	offsets(b, false)
	for r := 0; r < 4; r++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", r, r, r)
	}
	b.Label("sloop").Raw("CMP R4, $8").Raw("BLT sfold")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R3)(%s), VS%d", vidx(k), 4+k)
	}
	for k := 0; k < 4; k++ {
		b.XVADDDP(k, k, 4+k)
	}
	b.Raw("ADD $64, R3").Raw("ADD $-8, R4").Raw("BR sloop")
	b.Label("sfold")
	fold(b)
	b.Label("stail").Raw("CMP R4, $0").Raw("BEQ sdone").
		Raw("FMOVD 0(R3), F1").Raw("FADD F1, F0").
		Raw("ADD $8, R3").Raw("ADD $-1, R4").Raw("BR stail")
	b.Label("sdone").StoreRet("F0", "ret").Ret()
	return b.Func()
}

// dotKernel builds dotVSX(a, b *float64, n int) float64: four fused
// multiply-add accumulators (XVMADDADP), folded, plus a scalar tail.
func dotKernel() *emit.Function {
	sig := ppc64.Layout([]string{"a", "b", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64},
		[]string{"ret"}, []ppc64.Type{ppc64.Float64})
	b := ppc64.NewFunc("dotVSX", sig, 0)
	b.LoadArg("a", "R3").LoadArg("b", "R5").LoadArg("n", "R4")
	offsets(b, false)
	for r := 0; r < 4; r++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", r, r, r)
	}
	b.Label("dloop").Raw("CMP R4, $8").Raw("BLT dfold")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R3)(%s), VS%d", vidx(k), 32+k)
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		b.XVMADDADP(k, 32+k, 36+k)
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-8, R4").Raw("BR dloop")
	b.Label("dfold")
	fold(b)
	b.Label("dtail").Raw("CMP R4, $0").Raw("BEQ ddone").
		Raw("FMOVD 0(R3), F1").Raw("FMOVD 0(R5), F2").Raw("FMUL F2, F1").Raw("FADD F1, F0").
		Raw("ADD $8, R3").Raw("ADD $8, R5").Raw("ADD $-1, R4").Raw("BR dtail")
	b.Label("ddone").StoreRet("F0", "ret").Ret()
	return b.Func()
}

// sqrtKernel builds sqrtVSX(dst, src *float64, n int): XVSQRTDP over 8 doubles
// per iteration, FSQRT for the tail. Both are the correctly rounded IEEE
// square root, so the result is bit-identical to math.Sqrt.
func sqrtKernel() *emit.Function {
	sig := ppc64.Layout([]string{"dst", "src", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc("sqrtVSX", sig, 0)
	b.LoadArg("dst", "R3").LoadArg("src", "R5").LoadArg("n", "R4")
	offsets(b, false)
	b.Label("qloop").Raw("CMP R4, $8").Raw("BLT qtail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		b.XVSQRTDP(36+k, 32+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVD2X VS%d, (R3)(%s)", 36+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-8, R4").Raw("BR qloop")
	b.Label("qtail").Raw("CMP R4, $0").Raw("BEQ qdone").
		Raw("FMOVD 0(R5), F1").Raw("FSQRT F1, F1").Raw("FMOVD F1, 0(R3)").
		Raw("ADD $8, R3").Raw("ADD $8, R5").Raw("ADD $-1, R4").Raw("BR qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// binKernel builds name(dst, a, b *float64, n int): dst[i] = a[i] OP b[i], 8
// doubles per iteration with the VSX op and a scalar tail with sca. Each lane
// is one correctly rounded IEEE operation, bit-identical to the scalar oracle.
func binKernel(name, op, sca string) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "a", "b", "n"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R3").LoadArg("a", "R5").LoadArg("b", "R6").LoadArg("n", "R4")
	offsets(b, false)
	l := name[:3]
	b.Label(l+"loop").Raw("CMP R4, $8").Raw("BLT %stail", l)
	for k := 0; k < 4; k++ {
		b.Raw("LXVD2X (R5)(%s), VS%d", vidx(k), 32+k)
		b.Raw("LXVD2X (R6)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		switch op {
		case "add":
			b.XVADDDP(40+k, 32+k, 36+k)
		case "sub":
			b.XVSUBDP(40+k, 32+k, 36+k)
		case "mul":
			b.XVMULDP(40+k, 32+k, 36+k)
		case "div":
			b.XVDIVDP(40+k, 32+k, 36+k)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVD2X VS%d, (R3)(%s)", 40+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $64, R6").Raw("ADD $-8, R4").Raw("BR %sloop", l)
	b.Label(l+"tail").Raw("CMP R4, $0").Raw("BEQ %sdone", l).
		Raw("FMOVD 0(R5), F1").Raw("FMOVD 0(R6), F2").Raw("%s F2, F1", sca).Raw("FMOVD F1, 0(R3)").
		Raw("ADD $8, R3").Raw("ADD $8, R5").Raw("ADD $8, R6").Raw("ADD $-1, R4").Raw("BR %stail", l)
	b.Label(l + "done").Ret()
	return b.Func()
}

// gemmKernel builds gemmMicro8x8VSX(kc int, pa, pb, c *float64, ldc int):
//
//	for p in [0,kc):  C[r][col] += pa[p*8+r] * pb[p*8+col]   (r<8, col<8)
//
// with the scheme of OpenBLAS's POWER9 dgemm kernel: no splats. Each A pair
// (a_2p, a_2p+1) and its swap (one XXPERMDI) multiply each B column pair, so
// accumulator VS(8p+2k) holds the diagonal (C[2p][2k], C[2p+1][2k+1]) and
// VS(8p+2k+1) the anti-diagonal (C[2p+1][2k], C[2p][2k+1]); the store rebuilds
// the rows with XXPERMDI (DM=1). 32 fused multiply-adds per k step against 8
// loads and 4 swaps; two k steps per iteration, the second step's loads issued
// between the first step's FMAs. Measured on a POWER9 core (cfarm29): 16.9
// GFLOP/s, against 13.3 with LXVDSX splats, 14.8 with XXPERMDI splats and 15.4
// with those unrolled; OpenBLAS's own kernel reaches 23 there.
func gemmKernel() *emit.Function {
	name := "gemmMicro8x8VSX"
	sig := ppc64.Layout([]string{"kc", "pa", "pb", "c", "ldc"},
		[]ppc64.Type{ppc64.Int64, ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("kc", "R3").LoadArg("pa", "R4").LoadArg("pb", "R5").LoadArg("c", "R6").LoadArg("ldc", "R7")
	b.Raw("SLD $3, R7")
	offsets(b, false)
	for r := 0; r < 32; r++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", r, r, r)
	}
	b.Raw("CMP R3, $0").Raw("BEQ %sstore", name)
	// step uses B vectors bv..bv+3, A pairs av..av+3, swapped pairs sw..sw+3.
	loadsOf := func(bv, av int) []string {
		var l []string
		for k := 0; k < 4; k++ {
			l = append(l, fmt.Sprintf("LXVD2X (R5)(%s), VS%d", vidx(k), bv+k), fmt.Sprintf("LXVD2X (R4)(%s), VS%d", vidx(k), av+k))
		}
		return l
	}
	step := func(bv, av, sw int, pending []string) {
		for p := 0; p < 4; p++ {
			b.Raw("XXPERMDI VS%d, VS%d, $2, VS%d", av+p, av+p, sw+p)
		}
		i := 0
		for p := 0; p < 4; p++ {
			for k := 0; k < 4; k++ {
				b.XVMADDADP(8*p+2*k, av+p, bv+k)
				b.XVMADDADP(8*p+2*k+1, sw+p, bv+k)
				if i%2 == 1 && len(pending) > 0 {
					b.Raw("%s", pending[0])
					pending = pending[1:]
				}
				i++
			}
		}
		for _, l := range pending { // never leave a load out
			b.Raw("%s", l)
		}
	}
	b.Label(name+"loop2").Raw("CMP R3, $2").Raw("BLT %sone", name)
	for _, l := range loadsOf(32, 36) {
		b.Raw("%s", l)
	}
	b.Raw("ADD $64, R4").Raw("ADD $64, R5")
	step(32, 36, 40, loadsOf(44, 48))
	b.Raw("ADD $64, R4").Raw("ADD $64, R5")
	step(44, 48, 52, nil)
	b.Raw("ADD $-2, R3").Raw("BR %sloop2", name)
	b.Label(name+"one").Raw("CMP R3, $0").Raw("BEQ %sstore", name)
	for _, l := range loadsOf(32, 36) {
		b.Raw("%s", l)
	}
	step(32, 36, 40, nil)
	b.Label(name + "store")
	for p := 0; p < 4; p++ {
		for half := 0; half < 2; half++ { // row 2p, then 2p+1
			for k := 0; k < 4; k++ {
				d, a := 8*p+2*k, 8*p+2*k+1
				if half == 0 {
					b.Raw("XXPERMDI VS%d, VS%d, $1, VS%d", d, a, 56) // [diag.dw0, anti.dw1]
				} else {
					b.Raw("XXPERMDI VS%d, VS%d, $1, VS%d", a, d, 56) // [anti.dw0, diag.dw1]
				}
				b.Raw("LXVD2X (R6)(%s), VS%d", vidx(k), 57)
				b.XVADDDP(57, 57, 56)
				b.Raw("STXVD2X VS%d, (R6)(%s)", 57, vidx(k))
			}
			if !(p == 3 && half == 1) {
				b.Raw("ADD R7, R6")
			}
		}
	}
	b.Ret()
	return b.Func()
}
