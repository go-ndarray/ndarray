// Package blasmm is the row-major strided matrix product of ndarray on top of
// the general-stride GEMM of github.com/go-blas/blas (DgemmStrided and
// SgemmStrided), the one implementation of the packed, cache-blocked, multicore
// kernel.
//
// An operand is read through its row and column strides, as go-blas packs it,
// so a transposed or sliced view (a.T, a[:, ::2]) is never copied. go-blas
// takes strides of at least 1, and ndarray's views can hold others, which are
// normalised here: a stride of an axis of extent 1 is never used, so it is set
// to 1; an operand with a zero (broadcast) or negative (reversed) stride on an
// axis longer than 1 is copied to a contiguous matrix first, an O(rows*cols)
// pass beside the O(m*n*k) product. The result is written row-major, contiguous.
package blasmm

import "github.com/go-blas/blas"

// Mat is a strided matrix operand: element (i, j) is Data[Off+i*RS+j*CS].
type Mat[T float32 | float64] struct {
	Data   []T
	Off    int
	RS, CS int
}

// HasGemm32 reports whether SgemmStrided runs on a float32 SIMD kernel here;
// without one the caller multiplies in float64 instead.
func HasGemm32() bool { return blas.HasSgemmKernel() }

// MatMul writes the m x k by k x n product a*b into dst (m*n elements, row
// major, overwritten).
func MatMul(dst []float64, a, b Mat[float64], m, k, n int) {
	matMul(blas.DgemmStrided, dst, a, b, m, k, n)
}

// MatMul32 is MatMul in float32. It requires HasGemm32.
func MatMul32(dst []float32, a, b Mat[float32], m, k, n int) {
	matMul(blas.SgemmStrided, dst, a, b, m, k, n)
}

type stridedGemm[T float32 | float64] func(m, n, k int, alpha T, a []T, rsa, csa int, b []T, rsb, csb int, beta T, c []T, rsc, csc int) error

func matMul[T float32 | float64](gemm stridedGemm[T], dst []T, a, b Mat[T], m, k, n int) {
	if m == 0 || n == 0 {
		return
	}
	ad, rsa, csa := operand(a, m, k)
	bd, rsb, csb := operand(b, k, n)
	// The operands are valid views (the callers checked their shapes), so a
	// refusal here is a defect in this package, not an input error.
	if err := gemm(m, n, k, 1, ad, rsa, csa, bd, rsb, csb, 0, dst[:m*n], n, 1); err != nil {
		panic("ndarray: blasmm: " + err.Error())
	}
}

// operand returns x (rows x cols) as a slice and strides go-blas accepts.
func operand[T float32 | float64](x Mat[T], rows, cols int) (d []T, rs, cs int) {
	if rows == 0 || cols == 0 {
		return nil, 1, 1
	}
	rs, cs = x.RS, x.CS
	if rows == 1 {
		rs = 1 // never used
	}
	if cols == 1 {
		cs = 1
	}
	if rs >= 1 && cs >= 1 {
		return x.Data[x.Off:], rs, cs
	}
	d = make([]T, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			d[i*cols+j] = x.Data[x.Off+i*x.RS+j*x.CS]
		}
	}
	return d, cols, 1
}
