// Command gen produces sum_s390x.s, the vector-facility float64 kernels for
// s390x, via go-asmgen. Run with: go run . (or `go generate` from the kernels
// package).
//
// Go's s390x assembler names the instructions (VL/VLM/VSTM, VFADB, VFSDB,
// VFMDB, VFDDB, VFSQDB, VFMADB, VREPG, WFADB), so nothing is a WORD. A vector
// register holds two float64; each loop does eight (four registers) and a
// scalar tail does the rest. The Go side checks the vector facility in
// AT_HWCAP.
//
// Operand order, measured by running each instruction on known inputs under
// s390x emulation (the IBM manuals give the other order): VFSDB V1, V2, V3 is
// V3 = V2 - V1, VFDDB V1, V2, V3 is V3 = V2 / V1, VFMADB V1, V2, V3, V4 is
// V4 = V1*V2 + V3, and FSUB F1, F0 is F0 = F0 - F1.
//
// R1..R4 hold arguments and pointers (R0 cannot be a base register); V16..V19
// are the accumulators; F0..F15 alias the first element of V0..V15.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	f.Add(reduceKernel("sumVX", false))
	f.Add(reduceKernel("dotVX", true))
	f.Add(sqrtKernel())
	for _, k := range []struct{ name, vec, sca string }{
		{"addVX", "VFADB", "FADD"}, {"subVX", "VFSDB", "FSUB"},
		{"mulVX", "VFMDB", "FMUL"}, {"divVX", "VFDDB", "FDIV"},
	} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	if err := os.WriteFile("sum_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_s390x.s")
}

// reduceKernel builds NAME(a[, b] *float64, n int) float64: the sum of a, or
// with dot the sum of a[i]*b[i]. Four two-lane accumulators, folded at the end
// into F0, then the tail added one element at a time.
func reduceKernel(name string, dot bool) *emit.Function {
	args, types := []string{"a", "n"}, []s390x.Type{s390x.Ptr, s390x.Int64}
	if dot {
		args, types = []string{"a", "b", "n"}, []s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}
	}
	sig := s390x.Layout(args, types, []string{"ret"}, []s390x.Type{s390x.Float64})
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("a", "R1")
	if dot {
		b.LoadArg("b", "R3")
	}
	b.LoadArg("n", "R2")
	for r := 16; r < 20; r++ {
		b.Raw("VZERO V%d", r)
	}
	b.Raw("loop:").Raw("CMP R2, $8").Raw("BLT fold").
		Raw("VLM (R1), V0, V3")
	if dot {
		b.Raw("VLM (R3), V4, V7")
		for r := 0; r < 4; r++ {
			b.Raw("VFMADB V%d, V%d, V%d, V%d", r, 4+r, 16+r, 16+r) // acc += a*b
		}
		b.Raw("ADD $64, R3")
	} else {
		for r := 0; r < 4; r++ {
			b.Raw("VFADB V%d, V%d, V%d", r, 16+r, 16+r) // acc += a
		}
	}
	b.Raw("ADD $64, R1").Raw("SUB $8, R2").Raw("BR loop").
		Raw("fold:").
		Raw("VFADB V17, V16, V16").
		Raw("VFADB V19, V18, V18").
		Raw("VFADB V18, V16, V0").
		Raw("VREPG $1, V0, V1").
		Raw("WFADB V0, V1, V0"). // F0 = lane 0 + lane 1
		Raw("tail:").Raw("CMP R2, $0").Raw("BEQ done").
		Raw("FMOVD (R1), F1")
	if dot {
		b.Raw("FMOVD (R3), F2").Raw("FMUL F2, F1").Raw("ADD $8, R3")
	}
	b.Raw("FADD F1, F0").
		Raw("ADD $8, R1").Raw("SUB $1, R2").Raw("BR tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}

// sqrtKernel builds sqrtVX(dst, src *float64, n int).
func sqrtKernel() *emit.Function {
	sig := s390x.Layout([]string{"dst", "src", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("sqrtVX", sig, 0)
	b.LoadArg("dst", "R1").LoadArg("src", "R2").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $8").Raw("BLT tail").
		Raw("VLM (R2), V0, V3")
	for r := 0; r < 4; r++ {
		b.Raw("VFSQDB V%d, V%d", r, r)
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("SUB $8, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVD (R2), F0").Raw("FSQRT F0, F0").Raw("FMOVD F0, (R1)").
		Raw("ADD $8, R1").Raw("ADD $8, R2").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}

// binKernel builds NAME(dst, a, b *float64, n int): dst = a OP b. vec computes
// Vd = Vn OP Vm from "vec Vm, Vn, Vd"; sca computes F0 = F0 OP F1 from
// "sca F1, F0".
func binKernel(name, vec, sca string) *emit.Function {
	sig := s390x.Layout([]string{"dst", "a", "b", "n"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R1").LoadArg("a", "R2").LoadArg("b", "R3").LoadArg("n", "R4")
	b.Raw("loop:").Raw("CMP R4, $8").Raw("BLT tail").
		Raw("VLM (R2), V0, V3"). // a
		Raw("VLM (R3), V4, V7")  // b
	for r := 0; r < 4; r++ {
		b.Raw("%s V%d, V%d, V%d", vec, 4+r, r, r) // V_r = a OP b
	}
	b.Raw("VSTM V0, V3, (R1)").
		Raw("ADD $64, R1").Raw("ADD $64, R2").Raw("ADD $64, R3").Raw("SUB $8, R4").Raw("BR loop").
		Raw("tail:").Raw("CMP R4, $0").Raw("BEQ done").
		Raw("FMOVD (R2), F0").Raw("FMOVD (R3), F1").
		Raw(sca + " F1, F0").
		Raw("FMOVD F0, (R1)").
		Raw("ADD $8, R1").Raw("ADD $8, R2").Raw("ADD $8, R3").Raw("SUB $1, R4").Raw("BR tail").
		Raw("done:").Ret()
	return b.Func()
}
