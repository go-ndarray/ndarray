package kernels

import (
	"runtime"
	"sync"
)

// Packed, cache-blocked GEMM (the OpenBLAS/BLIS structure).
//
// The earlier register-blocked kernel read A and B straight from the source
// matrices, so on power-of-two row strides its four destination rows collided in
// the L1 cache and it lost to the autovectorized ikj kernel for large n. The fix
// is the technique every tuned BLAS uses: *packing*. A is copied into MR-tall row
// panels and B into NR-wide column panels in contiguous, unit-stride scratch, so
// the micro-kernel always streams conflict-free memory regardless of the source
// matrices' strides. A cache-blocking loop nest (NC over columns, KC over the
// contraction, MC over rows) keeps the packed B panel resident in L2/L3 and each
// packed A panel in L1, and a register-blocked SIMD-FMA micro-kernel
// (gemmMicro, arch-specific: NEON 4x8 on arm64, SSE2 4x4 on amd64, scalar 4x4
// elsewhere) does the inner work. Parallelised across row bands.
//
// The arithmetic is the standard ikj accumulation dst[i][j] += a[i][p]*b[p][j];
// the packed layout only relocates the operands, it does not reorder the sum, so
// the result is bit-identical to the scalar oracle (validated in CI and against
// numpy's A@B).

// Block sizes for the cache-blocking loop nest. They are vars so tests can pin
// them (forcing the multi-block paths on small inputs) and so they can be tuned
// per machine. Defaults are measured on the arm64 bench VM (Apple silicon):
//   - KC*NC packed-B panel (256*512*8 = 1 MiB) stays L2-resident,
//   - MC*KC packed-A panel (256*256*8 = 512 KiB) feeds the micro-kernel,
//   - they divide evenly into MR/NR tiles at the edges via zero padding.
var (
	blockMC = 256
	blockKC = 256
	blockNC = 512
)

// packBuf is one worker's reusable pair of pack buffers: paBuf for the MC*KC A
// panel, pbBuf for the KC*NC B panel. Pooling them keeps the per-call GEMM
// allocation-free above the goroutine launch, which matters at small n.
type packBuf struct {
	pa []float64
	pb []float64
}

// roundUp returns x rounded up to the next multiple of m (m >= 1).
func roundUp(x, m int) int { return (x + m - 1) / m * m }

// paCap / pbCap are the pack-buffer capacities. The A buffer holds
// ceil(MC/MR)*MR-tall panels (each MC row block is tiled into MR-tall panels and
// the last, partial panel is zero-padded to a full MR), so its row extent must be
// rounded UP to a multiple of MR — otherwise the bottom edge panel of an MC block
// whose height is not a multiple of MR would write past the buffer. Likewise the
// B buffer rounds the column extent up to a multiple of NR. (With the old MR=4
// and MC=256 the rounding was a no-op since 4 | 256; MR=6 makes it load-bearing.)
func paCap() int { return roundUp(blockMC, MR) * blockKC }
func pbCap() int { return blockKC * roundUp(blockNC, NR) }

var packPool = sync.Pool{New: func() any {
	return &packBuf{
		pa: make([]float64, paCap()),
		pb: make([]float64, pbCap()),
	}
}}

// getPackBuf returns a pooled buffer pair sized for the current block params. If
// the pooled buffers are too small for (possibly test-tuned) larger blocks it
// reallocates them, so a test that raises blockMC/KC/NC still gets valid scratch.
func getPackBuf() *packBuf {
	b := packPool.Get().(*packBuf)
	if cap(b.pa) < paCap() {
		b.pa = make([]float64, paCap())
	}
	if cap(b.pb) < pbCap() {
		b.pb = make([]float64, pbCap())
	}
	b.pa = b.pa[:paCap()]
	b.pb = b.pb[:pbCap()]
	return b
}

// packGemmRows computes the row band [r0,r1) of dst = a(m x k) * b(k x n) with
// the packed, cache-blocked algorithm, using buf for scratch. dst is the FULL
// (m x n) destination (the band writes only rows [r0,r1)); a and b are the full
// operands. dst must be pre-zeroed by the caller. The band must be MR-aligned at
// r0 (the parallel splitter guarantees this) so packed A panels never straddle a
// band boundary; r1 may be m.
func packGemmRows(r0, r1 int, dst, a, b []float64, k, n int, buf *packBuf) {
	pa, pb := buf.pa, buf.pb
	for jc := 0; jc < n; jc += blockNC { // L3 column block of B
		nc := min(blockNC, n-jc)
		for pc := 0; pc < k; pc += blockKC { // L2 contraction block
			kc := min(blockKC, k-pc)
			packB(b, n, pc, kc, jc, nc, pb) // pack B(kc x nc) -> NR-wide panels
			for ic := r0; ic < r1; ic += blockMC { // L1 row block of A
				mc := min(blockMC, r1-ic)
				packA(a, k, ic, mc, pc, kc, pa) // pack A(mc x kc) -> MR-tall panels
				for jr := 0; jr < nc; jr += NR {
					nr := min(NR, nc-jr)
					for ir := 0; ir < mc; ir += MR {
						mr := min(MR, mc-ir)
						c0 := (ic+ir)*n + jc + jr
						if mr == MR && nr == NR {
							gemmMicro(kc, pa[ir*kc:], pb[jr*kc:], dst[c0:], n)
						} else {
							gemmEdge(kc, pa[ir*kc:], pb[jr*kc:], dst, c0, n, mr, nr)
						}
					}
				}
			}
		}
	}
}

// packB copies B[pc:pc+kc][jc:jc+nc] into NR-wide column panels: for each panel
// jr the layout is pb[jr*kc + p*NR + c] = B[(pc+p)*ldb + jc+jr+c], with the c >=
// nr columns of an edge panel zero-filled so the micro-kernel reads a full NR.
func packB(b []float64, ldb, pc, kc, jc, nc int, pb []float64) {
	for jr := 0; jr < nc; jr += NR {
		nr := min(NR, nc-jr)
		dst := pb[jr*kc:]
		for p := 0; p < kc; p++ {
			srcRow := b[(pc+p)*ldb+jc+jr:]
			d := dst[p*NR:]
			for c := 0; c < nr; c++ {
				d[c] = srcRow[c]
			}
			for c := nr; c < NR; c++ {
				d[c] = 0
			}
		}
	}
}

// packA copies A[ic:ic+mc][pc:pc+kc] into MR-tall row panels: for each panel ir
// the layout is pa[ir*kc + p*MR + r] = A[(ic+ir+r)*lda + pc+p], with the r >= mr
// rows of an edge panel zero-filled so the micro-kernel reads a full MR.
func packA(a []float64, lda, ic, mc, pc, kc int, pa []float64) {
	for ir := 0; ir < mc; ir += MR {
		mr := min(MR, mc-ir)
		dst := pa[ir*kc:]
		for p := 0; p < kc; p++ {
			d := dst[p*MR:]
			for r := 0; r < mr; r++ {
				d[r] = a[(ic+ir+r)*lda+pc+p]
			}
			for r := mr; r < MR; r++ {
				d[r] = 0
			}
		}
	}
}

// gemmEdge is the scalar fallback for a partial (mr < MR or nr < NR) tile: it
// does the same dst[c0 + r*ldc + c] += sum_p pa[p*MR+r]*pb[p*NR+c] the micro-
// kernel does, reading the already-packed (and zero-padded) panels, so it stays
// contiguous and produces the identical ikj-order result. Only the matrix's
// ragged right/bottom edges take this path.
func gemmEdge(kc int, pa, pb, dst []float64, c0, ldc, mr, nr int) {
	for r := 0; r < mr; r++ {
		for c := 0; c < nr; c++ {
			var s float64
			for p := 0; p < kc; p++ {
				s += pa[p*MR+r] * pb[p*NR+c]
			}
			dst[c0+r*ldc+c] += s
		}
	}
}

// GemmThreshold is the minimum number of result elements (m*n) at which the
// packed GEMM fans out across goroutines. Small products run single-threaded
// (one worker, one pooled buffer) so the goroutine launch never dominates.
var GemmThreshold = 1 << 14

// MatMulP computes dst = a(m x k) * b(k x n) with the packed, cache-blocked GEMM,
// fanning the m output rows across cores above GemmThreshold. Each worker owns a
// disjoint, MR-aligned band of rows and writes a non-overlapping region of dst
// (b is read-only), so the result is identical to the serial computation. dst
// must be zeroed by the caller.
func MatMulP(dst, a, b []float64, m, k, n int) {
	if m*n < GemmThreshold {
		buf := getPackBuf()
		packGemmRows(0, m, dst, a, b, k, n, buf)
		packPool.Put(buf)
		return
	}
	w := runtime.GOMAXPROCS(0)
	if w > m {
		w = m
	}
	// Split the m rows into w near-equal bands, each rounded up to a multiple of
	// MR so no packed A panel straddles a band boundary; the last band absorbs
	// the remainder. This yields at most w non-empty bands.
	band := (m + w - 1) / w
	band = ((band + MR - 1) / MR) * MR
	var wg sync.WaitGroup
	for r0 := 0; r0 < m; r0 += band {
		r1 := min(r0+band, m)
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			buf := getPackBuf()
			packGemmRows(r0, r1, dst, a, b, k, n, buf)
			packPool.Put(buf)
		}(r0, r1)
	}
	wg.Wait()
}

// MatMul computes dst = a(m x k) * b(k x n) serially with the packed GEMM (the
// single-worker path of MatMulP). dst must be zeroed by the caller. It is kept as
// the named serial entry point used by callers and tests that want the exact
// non-parallel computation.
func MatMul(dst, a, b []float64, m, k, n int) {
	buf := getPackBuf()
	packGemmRows(0, m, dst, a, b, k, n, buf)
	packPool.Put(buf)
}

// dotRange returns sum(a[i]*b[i]) over equal-length slices, using four
// independent accumulators so the compiler keeps the FMA chain unrolled and (on
// amd64/arm64) auto-vectorises the multiply-add. The four-way grouping is a fixed
// reassociation of the sum; MatVecP/Dot1DP document the ULP trade-off where it is
// observable. It is the contiguous building block for both mat·vec and the 1-D
// dot — neither needs to go through the packing GEMM.
func dotRange(a, b []float64) float64 {
	var s0, s1, s2, s3 float64
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// MatVecP computes dst = a(m x k) * v(k,) as m independent contiguous row-dots,
// fanned across cores above GemmThreshold. Each output dst[i] is the dot of
// row i of a (the contiguous slice a[i*k:(i+1)*k]) with v, so the access pattern
// is pure unit-stride streaming — far cheaper than routing a single-column GEMM
// through the packer. dst need not be pre-zeroed (each entry is written, not
// accumulated). Rows are disjoint, so the parallel result equals the serial one.
func MatVecP(dst, a, v []float64, m, k int) {
	body := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = dotRange(a[i*k:i*k+k], v)
		}
	}
	if m*k < GemmThreshold {
		body(0, m)
		return
	}
	w := runtime.GOMAXPROCS(0)
	if w > m {
		w = m
	}
	parallelFor(m, w, body)
}

// VecMatP computes dst = v(k,) * a(k x n) = column sums weighted by v, i.e.
// dst[j] = sum_p v[p]*a[p*n+j]. It accumulates row by row so the inner loop over
// j streams a contiguous a row and the contiguous dst, which is the cache-
// friendly traversal (the transpose-free analogue of mat·vec for the 1-D·2-D
// case). dst must be zeroed by the caller.
func VecMatP(dst, v, a []float64, k, n int) {
	for p := 0; p < k; p++ {
		vp := v[p]
		row := a[p*n : p*n+n]
		for j := 0; j < n; j++ {
			dst[j] += vp * row[j]
		}
	}
}

// Dot1DP returns the inner product of two equal-length contiguous vectors,
// fanned across cores above ParThreshold: each worker dot-products its chunk with
// the unrolled dotRange and the partials are summed. The chunked four-way
// grouping reassociates the sum (a few ULP, like SumP / numpy pairwise); callers
// needing the exact left-to-right value use the scalar Dot1D.
func Dot1DP(a, b []float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return dotRange(a, b)
	}
	w := numWorkers(n)
	partials := make([]float64, w)
	chunk := (n + w - 1) / w
	parallelFor(w, w, func(lo, hi int) {
		for idx := lo; idx < hi; idx++ {
			s := idx * chunk
			if s >= n {
				// chunk rounding can leave trailing workers past the end (e.g.
				// many workers over a short, threshold-forced input); their
				// partial is just the additive identity.
				partials[idx] = 0
				continue
			}
			e := s + chunk
			if e > n {
				e = n
			}
			partials[idx] = dotRange(a[s:e], b[s:e])
		}
	})
	return Sum(partials)
}
