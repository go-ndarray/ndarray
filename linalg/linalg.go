// Package linalg is numpy.linalg for ndarray arrays, in pure Go: solving,
// inverting, determinants, the Cholesky, QR, singular-value and eigenvalue
// decompositions, least squares, pseudo-inverses, ranks, norms and condition
// numbers.
//
// Every function takes arrays of any dtype and treats the last two axes as
// the matrix, so a (..., M, N) input is a stack of matrices processed one by
// one (in parallel), as numpy's gufuncs do. As in numpy, the computation is
// always in double precision: float64 and complex128 natively, integers and
// bools after promotion to float64, float32 and complex64 after widening —
// their results are rounded back to float32 or complex64, as numpy's are.
//
// Decompositions follow LAPACK's algorithms (the ones numpy calls) and its
// sign and ordering conventions: LU with partial pivoting, Householder QR,
// Cholesky, Householder tridiagonalisation with implicit QL/QR for Hermitian
// eigenproblems, Golub–Kahan bidiagonalisation with implicit-shift QR for the
// SVD, and balancing, Hessenberg reduction, Francis QR and back-substitution
// for general eigenproblems. Large blocks are updated with the ndarray
// package's packed SIMD GEMM. docs/linalg.md describes the algorithms, their
// accuracy and their speed against numpy.
//
// Errors: numerical failures — a singular matrix, a matrix that is not
// positive definite, an iteration that does not converge — wrap ErrLinAlg,
// numpy's LinAlgError. Operands of the wrong rank or shape give ErrShape, and
// invalid options ErrArgs.
package linalg

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/go-ndarray/ndarray"
)

// ErrLinAlg is numpy's LinAlgError: the computation failed numerically. The
// more specific ErrSingular, ErrNotPosDef and ErrNoConvergence wrap it.
var ErrLinAlg = errors.New("linalg: LinAlgError")

// The numerical failures, each wrapping ErrLinAlg.
var (
	ErrSingular      = fmt.Errorf("%w: singular matrix", ErrLinAlg)
	ErrNotPosDef     = fmt.Errorf("%w: matrix is not positive definite", ErrLinAlg)
	ErrNoConvergence = fmt.Errorf("%w: did not converge", ErrLinAlg)
)

// ErrShape is returned for operands whose rank or shape the operation cannot
// take: fewer than two dimensions, a non-square matrix where a square one is
// needed, mismatched or non-broadcastable dimensions, or an empty matrix
// where numpy refuses one. (numpy raises LinAlgError for some of these and
// ValueError for others.)
var ErrShape = errors.New("linalg: invalid shape")

// ErrArgs is returned for invalid options: an unknown norm order, repeated
// or out-of-range axes, too few arrays.
var ErrArgs = errors.New("linalg: invalid argument")

// UPLO selects which triangle of a Hermitian matrix is read.
type UPLO uint8

// The two triangles. Lower is numpy's default.
const (
	Lower UPLO = iota
	Upper
)

// operand is an input whose last two axes are the matrix.
type operand struct {
	a      *ndarray.Array
	batch  []int // the leading dimensions
	m, n   int
	cplx   bool // computed in complex128 (else float64)
	single bool // results are float32 / complex64
}

// count is the number of matrices in the stack.
func (o operand) count() int { return prod(o.batch) }

func prod(s []int) int {
	p := 1
	for _, v := range s {
		p *= v
	}
	return p
}

// precision reports how numpy computes and returns a group of operands (its
// _commonType): complex if any is complex; single precision only if every
// one is float32 or complex64 — integers and bools count as double.
func precision(arrays ...*ndarray.Array) (cplx, single bool) {
	single = true
	for _, a := range arrays {
		dt := a.DType()
		cplx = cplx || dt.IsComplex()
		if dt != ndarray.Float32 && dt != ndarray.Complex64 {
			single = false
		}
	}
	return cplx, single
}

// stacked checks that a has at least two dimensions and describes it.
func stacked(a *ndarray.Array, fn string) (operand, error) {
	s := a.Shape()
	if len(s) < 2 {
		return operand{}, fmt.Errorf("%w: %s needs an array of at least two dimensions, got %d-D", ErrShape, fn, len(s))
	}
	cplx, single := precision(a)
	return operand{a: a, batch: s[:len(s)-2], m: s[len(s)-2], n: s[len(s)-1], cplx: cplx, single: single}, nil
}

// square is stacked for operations on square matrices.
func square(a *ndarray.Array, fn string) (operand, error) {
	o, err := stacked(a, fn)
	if err == nil && o.m != o.n {
		err = fmt.Errorf("%w: %s needs square matrices, got %d×%d", ErrShape, fn, o.m, o.n)
	}
	return o, err
}

// load returns a contiguous copy of a's elements converted to T.
func load[T scalar](a *ndarray.Array) []T {
	if isCplx[T]() {
		d, _ := ndarray.Data[complex128](a.AsType(ndarray.Complex128))
		return any(d).([]T)
	}
	d, _ := ndarray.Data[float64](a.AsType(ndarray.Float64))
	return any(d).([]T)
}

// wrap makes an array of the given shape from computed values, rounded to
// single precision when single is set.
func wrap[T scalar](d []T, shape []int, single bool) *ndarray.Array {
	if z, ok := any(d).([]complex128); ok {
		r, _ := ndarray.FromSlice(z, shape...)
		if single {
			r = r.AsType(ndarray.Complex64)
		}
		return r
	}
	r, _ := ndarray.FromSlice(any(d).([]float64), shape...)
	if single {
		r = r.AsType(ndarray.Float32)
	}
	return r
}

// shapeOf joins a batch shape and trailing dimensions.
func shapeOf(batch []int, tail ...int) []int {
	return append(append([]int(nil), batch...), tail...)
}

// parallelFor runs f(0..count-1), spreading the calls over the processors
// when there are several and each is worth a goroutine (work is the rough
// cost of one call in flops).
func parallelFor(count, work int, f func(i int)) {
	w := min(runtime.GOMAXPROCS(0), count)
	if w <= 1 || count*work < 1<<15 {
		for i := 0; i < count; i++ {
			f(i)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for g := 0; g < w; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= count {
					return
				}
				f(i)
			}
		}()
	}
	wg.Wait()
}

// firstErr collects the error of the lowest-indexed failing matrix of a
// parallel loop, so the error reported does not depend on scheduling.
type firstErr struct {
	mu  sync.Mutex
	idx int
	err error
}

func (f *firstErr) set(i int, err error) {
	f.mu.Lock()
	if f.err == nil || i < f.idx {
		f.idx, f.err = i, err
	}
	f.mu.Unlock()
}

// broadcast returns the broadcast of two batch shapes, numpy's rules.
func broadcast(x, y []int) ([]int, bool) {
	n := max(len(x), len(y))
	out := make([]int, n)
	for i := range out {
		a, b := dimAt(x, i-n+len(x)), dimAt(y, i-n+len(y))
		switch {
		case a == b || b == 1:
			out[i] = a
		case a == 1:
			out[i] = b
		default:
			return nil, false
		}
	}
	return out, true
}

func dimAt(s []int, i int) int {
	if i < 0 {
		return 1
	}
	return s[i]
}

// batchIndex maps the linear index of an element of the broadcast batch
// shape out to the linear index in the batch shape in that broadcasts to it.
func batchIndex(out, in []int, i int) int {
	idx, stride := 0, 1
	for ax := len(out) - 1; ax >= 0; ax-- {
		c := i % out[ax]
		i /= out[ax]
		if j := ax - len(out) + len(in); j >= 0 {
			if in[j] != 1 {
				idx += c * stride
			}
			stride *= in[j]
		}
	}
	return idx
}
