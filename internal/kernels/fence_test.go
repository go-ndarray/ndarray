//go:build (amd64 || arm64 || ppc64le || loong64 || riscv64) && (linux || darwin)

package kernels

import (
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// The assembly kernels take raw pointers and trust their Go wrappers for the
// lengths: an off-by-one in a tail loop, or a load issued one iteration early,
// reads or writes memory that is not the slice's. Go's bounds checks cannot see
// inside a .s file, and a stray access into the rest of the heap goes unnoticed.
//
// A fence makes it fault instead. Each operand is placed flush against an
// inaccessible page (PROT_NONE), once ending at the page and once starting
// right after one, so an access a single element past either end is a
// SIGSEGV. Prefetches are hints and do not fault, so the kernels' prefetching
// past the end is allowed, as the hardware allows it.

// fenced is a mapping of [guard | data | guard] whose data region holds at
// least n float64s.
type fenced struct {
	mem  []byte
	data []float64 // the whole region between the guards
}

func newFenced(t *testing.T, n int) *fenced {
	t.Helper()
	pg := os.Getpagesize()
	dataBytes := (n*8 + pg - 1) / pg * pg
	mem, err := syscall.Mmap(-1, 0, dataBytes+2*pg,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	for _, g := range [][]byte{mem[:pg], mem[pg+dataBytes:]} {
		if err := syscall.Mprotect(g, syscall.PROT_NONE); err != nil {
			t.Fatalf("mprotect: %v", err)
		}
	}
	d := unsafe.Slice((*float64)(unsafe.Pointer(&mem[pg])), dataBytes/8)
	for i := range d {
		d[i] = float64(i%7) + 1 // positive, so sqrt and div stay finite
	}
	return &fenced{mem: mem, data: d}
}

// head is n elements starting right after the leading guard; tail is n
// elements ending right before the trailing guard.
func (f *fenced) head(n int) []float64 { return f.data[:n:n] }
func (f *fenced) tail(n int) []float64 { return f.data[len(f.data)-n:] }

// fences returns, for n elements, the operand placements to try.
func fences(t *testing.T, n int) [][]float64 {
	f := newFenced(t, n)
	return [][]float64{f.head(n), f.tail(n)}
}

func TestKernelsStayInsideTheirSlices(t *testing.T) {
	for n := 1; n <= 70; n++ {
		for _, a := range fences(t, n) {
			for _, b := range fences(t, n) {
				for _, dst := range fences(t, n) {
					sumSIMD(a)
					maxSIMD(a)
					minSIMD(a)
					dotSIMD(a, b)
					sqrtSIMD(dst, a)
					addBin(dst, a, b)
					subBin(dst, a, b)
					mulBin(dst, a, b)
					divBin(dst, a, b)
				}
			}
		}
	}
}

func TestGemmMicroStaysInsideItsOperands(t *testing.T) {
	run := func() {
		for kc := 1; kc <= 9; kc++ {
			for _, ldc := range []int{NR, NR + 3, 64} {
				for _, pa := range fences(t, kc*MR) {
					for _, pb := range fences(t, kc*NR) {
						for _, dst := range fences(t, (MR-1)*ldc+NR) {
							gemmMicro(kc, pa, pb, dst, ldc)
						}
					}
				}
			}
		}
	}
	run()
	forEachFallback(run)
}
