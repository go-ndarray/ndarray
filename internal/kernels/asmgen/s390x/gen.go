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
	// The float64 sum, sqrt and Add/Sub/Mul/Div kernels live in go-simd/floats
	// (lanes_s390x_gen.go there) since floats v0.10.0.
	f.Add(extremumKernel("maxVXE", "VFMAXDB", "WFMAXDB"))
	f.Add(extremumKernel("minVXE", "VFMINDB", "WFMINDB"))
	f.Add(blocks8Kernel())
	// max32VXE/min32VXE moved to go-simd/floats (Float32Max, Float32Min;
	// go-ndarray/ndarray#98).
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

// extremumKernel builds NAME(a *float64, n int) float64, the maximum (or
// minimum) of a, for n >= 8. VFMAXDB/VFMINDB and their scalar WF forms come
// with the vector-enhancements facility 1 (z14), so the Go side gates these
// on AT_HWCAP's VXRS_EXT. Mode 1 is Java's Math.max/min: a NaN operand gives
// NaN, and -0 orders below +0, which is NumPy's and Go's builtin max/min rule.
// The operation is associative under it, so four two-lane accumulators seeded
// with the first eight elements, folded at the end, give the serial result.
func extremumKernel(name, vec, sca string) *emit.Function {
	sig := s390x.Layout([]string{"a", "n"}, []s390x.Type{s390x.Ptr, s390x.Int64},
		[]string{"ret"}, []s390x.Type{s390x.Float64})
	b := s390x.NewFunc(name, sig, 0)
	b.LoadArg("a", "R1").LoadArg("n", "R2").
		Raw("VLM (R1), V16, V19").
		Raw("ADD $64, R1").Raw("SUB $8, R2").
		Raw("loop:").Raw("CMP R2, $8").Raw("BLT fold").
		Raw("VLM (R1), V0, V3")
	for r := 0; r < 4; r++ {
		b.Raw("%s $1, V%d, V%d, V%d", vec, r, 16+r, 16+r)
	}
	b.Raw("ADD $64, R1").Raw("SUB $8, R2").Raw("BR loop").
		Raw("fold:").
		Raw("%s $1, V17, V16, V16", vec).
		Raw("%s $1, V19, V18, V18", vec).
		Raw("%s $1, V18, V16, V0", vec).
		Raw("VREPG $1, V0, V1").
		Raw("%s $1, F1, F0, F0", sca). // F0 = extremum of the two lanes
		Raw("tail:").Raw("CMP R2, $0").Raw("BEQ done").
		Raw("FMOVD (R1), F1").
		Raw("%s $1, F1, F0, F0", sca).
		Raw("ADD $8, R1").Raw("SUB $1, R2").Raw("BR tail").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}

// blocks8Kernel builds blocks8VXE(r, a *float32, blocks int): for each of the
// blocks of eight float32 at a, in order, r[j] += a[8i+j] (NumPy's pairwise-sum
// block). The eight accumulators are split over two vector registers, V0 and
// V1 (two dependency chains, as on the other targets), and VFASB is one IEEE
// float32 addition per lane, so the result is the scalar loop's bit for bit.
// VL/VST keep memory order on this big-endian machine, so every float keeps
// its lane. VFASB is the vector-enhancements facility 1 (z14); Go's assembler
// lacks it, and go-asmgen emits it as WORD+BYTE+BYTE pinned against GNU as
// (the cmd/asm spelling is CL 846286). blocks >= 1.
func blocks8Kernel() *emit.Function {
	sig := s390x.Layout([]string{"r", "a", "blocks"},
		[]s390x.Type{s390x.Ptr, s390x.Ptr, s390x.Int64}, nil, nil)
	b := s390x.NewFunc("blocks8VXE", sig, 0)
	b.LoadArg("r", "R1").LoadArg("a", "R2").LoadArg("blocks", "R3").
		Raw("VL 0(R1), V0").Raw("VL 16(R1), V1").
		Raw("b8loop:").
		Raw("VL 0(R2), V2").Raw("VL 16(R2), V3")
	b.VFASB(0, 0, 2).VFASB(1, 1, 3)
	b.Raw("ADD $32, R2").Raw("SUB $1, R3").Raw("CMPBNE R3, $0, b8loop").
		Raw("VST V0, 0(R1)").Raw("VST V1, 16(R1)").
		Ret()
	return b.Func()
}
