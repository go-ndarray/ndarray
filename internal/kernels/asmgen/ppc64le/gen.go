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
	f.Add(dotKernel())
	// The float64 sum, sqrt and Add/Sub/Mul/Div kernels live in go-simd/floats
	// (lanes_ppc64le_gen.go there) since floats v0.10.0.
	for _, k := range []struct {
		name, vec, sca string
	}{{"addVSX", "add", "FADD"}, {"subVSX", "sub", "FSUB"}, {"mulVSX", "mul", "FMUL"}, {"divVSX", "div", "FDIV"}} {
		f.Add(bin32Kernel(k.name[:3]+"32VSX", k.vec, k.sca+"S"))
	}
	f.Add(blocks8Kernel())
	f.Add(sqrt32Kernel())
	abs64 := f.Data("abs64", lanes(8, 0x7fffffffffffffff))
	inf64 := f.Data("inf64", lanes(8, 0x7ff0000000000000))
	// max32VSX/min32VSX moved to go-simd/floats (Float32Max, Float32Min;
	// go-ndarray/ndarray#98).
	f.Add(extremumKernel("maxVSX", 64, true, abs64, inf64))
	f.Add(extremumKernel("minVSX", 64, false, abs64, inf64))
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

// lanes is a 16-byte vector constant: v in every lane of size bytes, each lane
// little-endian, as LXVD2X/LXVW4X load it on ppc64le (every lane equal, so
// their lane order does not matter).
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
// positive multiple of the block (8 doubles or 16 floats). It reduces a with
// xvmaxdp/xvmindp (xvmaxsp/xvminsp) into four accumulators, folds them
// vertically into one vector stored at out, and returns non-zero if any
// element was NaN. The Go wrapper folds the lanes and the tail with Go's
// max/min. So neither the instruction's NaN rule nor its sign of a zero tie
// matters: NaN is detected separately, and a zero result goes to the scalar
// oracle.
//
// NaN detection is integer arithmetic on the VMX registers, because Go's
// assembler has no VSX float compare: a lane is NaN iff its magnitude bits
// (x AND abs) exceed +Inf's bits as an unsigned integer (vcmpgtud/vcmpgtuw).
// The masks are OR-accumulated in V22. Registers: VS0-VS3 accumulators;
// V4-V7 (VS36-VS39) the block; V8-V11 scratch; V20, V21 the abs and +Inf
// constants; V22 the NaN mask.
func extremumKernel(name string, bits int, isMax bool, abs, inf string) *emit.Function {
	sig := ppc64.Layout([]string{"a", "n", "out"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Int64, ppc64.Ptr}, []string{"ret"}, []ppc64.Type{ppc64.Int64})
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R3").LoadArg("n", "R4").LoadArg("out", "R5")
	offsets(b, false)
	ld, st, cmp, block := "LXVD2X", "STXVD2X", "VCMPGTUD", 8
	op := b.XVMAXDP
	if !isMax {
		op = b.XVMINDP
	}
	if bits == 32 {
		ld, st, cmp, block = "LXVW4X", "STXVW4X", "VCMPGTUW", 16
		op = b.XVMAXSP
		if !isMax {
			op = b.XVMINSP
		}
	}
	b.Raw("MOVD $%s+0(SB), R7", abs).Raw("%s (R7)(R0), VS52", ld).
		Raw("MOVD $%s+0(SB), R7", inf).Raw("%s (R7)(R0), VS53", ld).
		Raw("XXLXOR VS54, VS54, VS54")
	loadAndScan := func() {
		for k := 0; k < 4; k++ {
			b.Raw("%s (R3)(%s), VS%d", ld, vidx(k), 36+k)
		}
		for k := 0; k < 4; k++ {
			b.Raw("VAND V%d, V20, V%d", 4+k, 8+k).
				Raw("%s V%d, V21, V%d", cmp, 8+k, 8+k). // magnitude > +Inf: NaN
				Raw("VOR V%d, V22, V22", 8+k)
		}
		b.Raw("ADD $64, R3").Raw("ADD $-%d, R4", block)
	}
	// The first block seeds the accumulators.
	loadAndScan()
	for k := 0; k < 4; k++ {
		b.Raw("XXLOR VS%d, VS%d, VS%d", 36+k, 36+k, k)
	}
	b.Label(name+"loop").Raw("CMP R4, $0").Raw("BEQ %sfold", name)
	loadAndScan()
	for k := 0; k < 4; k++ {
		op(k, k, 36+k)
	}
	b.Raw("BR %sloop", name)
	b.Label(name + "fold")
	op(0, 0, 1)
	op(2, 2, 3)
	op(0, 0, 2)
	b.Raw("%s VS0, (R5)(R0)", st).
		// The NaN mask's two doublewords, ORed into R6.
		Raw("XXPERMDI VS54, VS54, $2, VS55").
		Raw("MFVSRD VS54, R6").Raw("MFVSRD VS55, R7").Raw("OR R7, R6").
		StoreRet("R6", "ret").Ret()
	return b.Func()
}

// sqrt32Kernel builds sqrt32VSX(dst, src *float32, n int): xvsqrtsp over 16
// floats per iteration, FSQRTS for the tail; the correctly rounded float32
// square root, bit-identical to float32(math.Sqrt(float64(x))).
func sqrt32Kernel() *emit.Function {
	sig := ppc64.Layout([]string{"dst", "src", "n"}, []ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc("sqrt32VSX", sig, 0)
	b.LoadArg("dst", "R3").LoadArg("src", "R5").LoadArg("n", "R4")
	offsets(b, false)
	b.Label("qloop").Raw("CMP R4, $16").Raw("BLT qtail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVW4X (R5)(%s), VS%d", vidx(k), 32+k)
	}
	for k := 0; k < 4; k++ {
		b.XVSQRTSP(36+k, 32+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVW4X VS%d, (R3)(%s)", 36+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $-16, R4").Raw("BR qloop")
	b.Label("qtail").Raw("CMP R4, $0").Raw("BEQ qdone").
		Raw("FMOVS 0(R5), F1").Raw("FSQRTS F1, F1").Raw("FMOVS F1, 0(R3)").
		Raw("ADD $4, R3").Raw("ADD $4, R5").Raw("ADD $-1, R4").Raw("BR qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// bin32Kernel builds name(dst, a, b *float32, n int): dst[i] = a[i] OP b[i],
// 16 floats per iteration with the VSX single-precision op, a scalar tail with
// sca. LXVW4X/STXVW4X put each word at the same lane in every operand and back
// at its own index. Each lane is one IEEE float32 operation, bit-identical to
// the scalar oracle.
func bin32Kernel(name, op, sca string) *emit.Function {
	sig := ppc64.Layout([]string{"dst", "a", "b", "n"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R3").LoadArg("a", "R5").LoadArg("b", "R6").LoadArg("n", "R4")
	offsets(b, false)
	b.Label("floop").Raw("CMP R4, $16").Raw("BLT ftail")
	for k := 0; k < 4; k++ {
		b.Raw("LXVW4X (R5)(%s), VS%d", vidx(k), 32+k)
		b.Raw("LXVW4X (R6)(%s), VS%d", vidx(k), 36+k)
	}
	for k := 0; k < 4; k++ {
		switch op {
		case "add":
			b.XVADDSP(40+k, 32+k, 36+k)
		case "sub":
			b.XVSUBSP(40+k, 32+k, 36+k)
		case "mul":
			b.XVMULSP(40+k, 32+k, 36+k)
		case "div":
			b.XVDIVSP(40+k, 32+k, 36+k)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("STXVW4X VS%d, (R3)(%s)", 40+k, vidx(k))
	}
	b.Raw("ADD $64, R3").Raw("ADD $64, R5").Raw("ADD $64, R6").Raw("ADD $-16, R4").Raw("BR floop")
	b.Label("ftail").Raw("CMP R4, $0").Raw("BEQ fdone").
		Raw("FMOVS 0(R5), F1").Raw("FMOVS 0(R6), F2").Raw("%s F2, F1", sca).Raw("FMOVS F1, 0(R3)").
		Raw("ADD $4, R3").Raw("ADD $4, R5").Raw("ADD $4, R6").Raw("ADD $-1, R4").Raw("BR ftail")
	b.Label("fdone").Ret()
	return b.Func()
}

// blocks8Kernel builds blocks8VSX(r, a *float32, blocks int): for each of the
// blocks of eight float32 at a, in order, r[j] += a[8i+j] (NumPy's pairwise-sum
// block). The eight accumulators are split over two VSX registers, VS0 and VS1
// (two dependency chains, as SSE2 on amd64 and LSX on loong64), and XVADDSP is
// one IEEE float32 addition per lane, so the result is the scalar loop's bit
// for bit. LXVW4X/STXVW4X put each word at the same lane in the accumulators
// and the input, and back at its own index on the store. blocks >= 1.
func blocks8Kernel() *emit.Function {
	sig := ppc64.Layout([]string{"r", "a", "blocks"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Ptr, ppc64.Int64}, nil, nil)
	b := ppc64.NewFunc("blocks8VSX", sig, 0)
	b.LoadArg("r", "R3").LoadArg("a", "R4").LoadArg("blocks", "R5")
	b.Raw("MOVD $16, R14")
	b.Raw("LXVW4X (R3)(R0), VS0").Raw("LXVW4X (R3)(R14), VS1")
	b.Label("b8loop").
		Raw("LXVW4X (R4)(R0), VS32").
		Raw("LXVW4X (R4)(R14), VS33")
	b.XVADDSP(0, 0, 32).XVADDSP(1, 1, 33)
	b.Raw("ADD $32, R4").Raw("ADD $-1, R5").Raw("CMP R5, $0").Raw("BNE b8loop")
	b.Raw("STXVW4X VS0, (R3)(R0)").Raw("STXVW4X VS1, (R3)(R14)")
	b.Ret()
	return b.Func()
}
