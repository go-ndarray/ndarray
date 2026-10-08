// Command gen produces sum_riscv64.s, the RVV (vector extension) float64
// kernels for riscv64, via go-asmgen. Run with: go run . (or `go generate`
// from the kernels package).
//
// Go's riscv64 assembler names the vector instructions these kernels need
// (VSETVLI, VLE64V/VSE64V, VFADDVV, VFSUBVV, VFMULVV, VFDIVVV, VFSQRTV,
// VFMACCVV, VFREDUSUMVS, VFMVSF/VFMVFS), so nothing is emitted as a WORD. The
// loops are vector-length agnostic: VSETVLI hands out as many elements as the
// hardware vector holds (LMUL=8 groups eight registers), including a short last
// strip, so there is no scalar tail. The V extension is optional, so the Go
// side calls these only when the kernel reports it (AT_HWCAP).
//
// Operand order, decoded from Go's own testdata encodings: VFADDVV V1, V2, V3
// computes V3 = V2 + V1 (likewise SUB and DIV: V3 = V2 - V1, V2 / V1);
// VFMACCVV V1, V2, V3 computes V3 += V2*V1; VFREDUSUMVS Vs, Vv, Vd computes
// Vd[0] = Vs[0] + sum(Vv).
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	f.Add(reduceKernel("sumRVV", false))
	f.Add(reduceKernel("dotRVV", true))
	f.Add(unaryKernel("sqrtRVV", "VFSQRTV V0, V16"))
	for _, k := range []struct{ name, op string }{
		{"addRVV", "VFADDVV"}, {"subRVV", "VFSUBVV"},
		{"mulRVV", "VFMULVV"}, {"divRVV", "VFDIVVV"},
	} {
		f.Add(binKernel(k.name, k.op))
	}
	if err := os.WriteFile("sum_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sum_riscv64.s")
}

// advance moves each pointer register past vl (in X) float64s.
func advance(b *riscv64.Builder, vl string, ptrs ...string) {
	b.Raw("SLLI $3, %s, X17", vl)
	for _, p := range ptrs {
		b.Raw("ADD X17, %s, %s", p, p)
	}
}

// reduceKernel builds NAME(a[, b] *float64, n int) float64 for n >= 1: the sum
// of a, or with dot the sum of a[i]*b[i]. A vector accumulator of VLMAX lanes
// is updated tail-undisturbed, so the lanes past a short last strip keep their
// partial sums, then reduced to one scalar.
func reduceKernel(name string, dot bool) *emit.Function {
	args, types := []string{"a", "n"}, []riscv64.Type{riscv64.Ptr, riscv64.Int64}
	if dot {
		args, types = []string{"a", "b", "n"}, []riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Int64}
	}
	sig := riscv64.Layout(args, types, []string{"ret"}, []riscv64.Type{riscv64.Float64})
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("a", "X10")
	n := "X11"
	if dot {
		b.LoadArg("b", "X11")
		n = "X12"
	}
	b.LoadArg("n", n)
	b.Raw("VSETVLI X0, E64, M8, TA, MA, X13"). // X13 = VLMAX
							Raw("VMVVI $0, V16") // accumulator = 0.0 in every lane
	b.Raw("loop:").
		Raw("VSETVLI %s, E64, M8, TU, MA, X13", n).
		Raw("VLE64V (X10), V0")
	if dot {
		b.Raw("VLE64V (X11), V8").Raw("VFMACCVV V8, V0, V16") // V16 += a*b
		advance(b, "X13", "X10", "X11")
	} else {
		b.Raw("VFADDVV V0, V16, V16") // V16 += a
		advance(b, "X13", "X10")
	}
	b.Raw("SUB X13, %s, %s", n, n).
		Raw("BNEZ %s, loop", n).
		Raw("VSETVLI X0, E64, M8, TA, MA, X13").
		Raw("FMVDX X0, F0").
		Raw("VFMVSF F0, V24").
		Raw("VFREDUSUMVS V24, V16, V24").
		Raw("VFMVFS V24, F0").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}

// unaryKernel builds NAME(dst, src *float64, n int) for n >= 1: op reads V0 and
// writes V16.
func unaryKernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "src", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("src", "X11").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M8, TA, MA, X14").
		Raw("VLE64V (X11), V0").
		Raw(op).
		Raw("VSE64V V16, (X10)")
	advance(b, "X14", "X10", "X11")
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}

// binKernel builds NAME(dst, a, b *float64, n int) for n >= 1: dst = a OP b.
func binKernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "a", "b", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("a", "X11").LoadArg("b", "X12").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M8, TA, MA, X14").
		Raw("VLE64V (X11), V0").
		Raw("VLE64V (X12), V8").
		Raw("%s V8, V0, V16", op). // V16 = a OP b
		Raw("VSE64V V16, (X10)")
	advance(b, "X14", "X10", "X11", "X12")
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}
