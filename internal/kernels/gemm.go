package kernels

import (
	"runtime"
	"sync"
	"sync/atomic"
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
// (gemmMicro, arch-specific: NEON 4x8 on arm64, AVX2/FMA 6x8 or SSE2 on amd64,
// scalar 4x4 elsewhere) does the inner work. Parallelised across row bands.
//
// The driver is generic over the element type: float64 everywhere (gemm64), and
// float32 where a float32 micro-kernel exists (gemm32: NEON 8x8 on arm64,
// AVX2/FMA 6x16 on amd64), which accumulates in float32 as a BLAS sgemm does.
// Each element type brings its own tile and its own pack-buffer pools.
//
// The arithmetic is the ikj accumulation dst[i][j] += a[i][p]*b[p][j]: within
// one KC block the products are summed in p order, and each block's sum is then
// added into dst. For k <= KC that is the scalar oracle's order exactly; above
// it the block partials regroup the sum, a valid reordering that can move the
// last place (as every blocked BLAS does). Kernels that fuse the multiply-add
// (arm64, amd64 with FMA) round once per step where an unfused one rounds
// twice. On data whose products and sums are exact (the integer-valued tests)
// every path equals the oracle bit for bit.

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

// bandTilesMR is the height, in MR-row micro-tiles, of one dynamic work band in
// MatMulP (so a band is bandTilesMR*MR rows). It is sized so the m rows dice into
// several bands per core: enough oversubscription for the atomic work-counter to
// balance the M-series P/E cores, while large enough that the per-band A-pack and
// goroutine-loop overhead stays amortized. A var so it can be tuned/pinned.
var bandTilesMR = 8

// gemmFloat is the element types the packed GEMM multiplies.
type gemmFloat interface{ float32 | float64 }

// MatOf is a read-only strided view of a matrix: element (i, j) is
// Data[Off + i*RS + j*CS]. Packing reads the operands through it, so a
// transposed, sliced or offset view is multiplied in place — the packing copy
// the GEMM makes anyway does the gather — instead of being materialised into
// a contiguous copy first (what BLAS's trans flags buy). Strides may be
// negative or zero; every (i, j) in range must index Data.
type MatOf[T gemmFloat] struct {
	Data        []T
	Off, RS, CS int
}

// Mat and Mat32 are the float64 and float32 operand views.
type (
	Mat   = MatOf[float64]
	Mat32 = MatOf[float32]
)

// rowMajor views a contiguous row-major matrix with ld elements per row.
func rowMajor[T gemmFloat](data []T, ld int) MatOf[T] {
	return MatOf[T]{Data: data, RS: ld, CS: 1}
}

// gemmKernel is one element type's micro-kernel, the mr x nr tile it computes,
// and the pools of pack buffers sized for that tile. micro adds the mr x nr tile
// sum_p pa[p*mr..]*pb[p*nr..] into dst, whose rows are ldc apart; see gemmMicro.
//
// microUA, where there is one, is the same tile with A read in place: rows
// lda > 0 elements apart from a[0], k steps along each row, for the top mr
// (1..mr) rows only. gemmBand then never packs a row-major A.
type gemmKernel[T gemmFloat] struct {
	mr, nr         int
	micro          func(kc int, pa, pb, dst []T, ldc int)
	microUA        func(kc int, a []T, lda int, pb, dst []T, ldc, mr int)
	paPool, pbPool sync.Pool
}

// unpackA lets the GEMM read a row-major A in place where the kernel can. It
// is a var so tests can drive the packed path on the same inputs.
var unpackA = true

// gemm64 is the float64 GEMM, on the per-architecture gemmMicro.
var gemm64 = &gemmKernel[float64]{mr: MR, nr: NR, micro: gemmMicro, microUA: gemmMicroUA}

// packBuf is a pair of pack buffers: pa for the MC*KC A panel, pb for the KC*NC
// B panel. The two come from separate pools, because the parallel GEMM needs
// one pa per worker but only two pb in all: drawing a pair per worker held
// 16 MiB of unused B buffers at 16 workers, which the garbage collector empties
// out of the pool and the next call zeroes again.
type packBuf[T gemmFloat] struct {
	pa []T
	pb []T
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
//
// The A buffer also carries, past the panels, the mr x nr scratch tile gemmEdge
// runs the kernel into. The tile cannot live on gemmEdge's stack: the kernel is
// called through a func value, so escape analysis would move it to the heap at
// every edge tile.
func (g *gemmKernel[T]) paCap() int { return roundUp(blockMC, g.mr)*blockKC + g.mr*g.nr }
func (g *gemmKernel[T]) pbCap() int { return blockKC * roundUp(blockNC, g.nr) }

// getPA and getPB return a pooled pack buffer sized for the current block
// parameters, allocating one when the pool is empty or its buffer is too small
// (a test may raise blockMC/KC/NC). putPA and putPB return it.
func (g *gemmKernel[T]) getPA() []T  { return getScratch[T](&g.paPool, g.paCap()) }
func (g *gemmKernel[T]) getPB() []T  { return getScratch[T](&g.pbPool, g.pbCap()) }
func (g *gemmKernel[T]) putPA(b []T) { g.paPool.Put(&b) }
func (g *gemmKernel[T]) putPB(b []T) { g.pbPool.Put(&b) }

func getScratch[T gemmFloat](p *sync.Pool, n int) []T {
	if b, ok := p.Get().(*[]T); ok && cap(*b) >= n {
		return (*b)[:n]
	}
	return make([]T, n)
}

// getPackBuf returns a pooled pair, for the single-worker paths.
func (g *gemmKernel[T]) getPackBuf() *packBuf[T] {
	return &packBuf[T]{pa: g.getPA(), pb: g.getPB()}
}

func (g *gemmKernel[T]) putPackBuf(b *packBuf[T]) {
	g.putPA(b.pa)
	g.putPB(b.pb)
}

// packGemmRows computes the row band [r0,r1) of dst = a(m x k) * b(k x n) with
// the packed, cache-blocked algorithm, using buf for scratch. dst is the FULL
// (m x n) destination (the band writes only rows [r0,r1)); a and b are the full
// operands. dst must be pre-zeroed by the caller. The band must be MR-aligned at
// r0 (the parallel splitter guarantees this) so packed A panels never straddle a
// band boundary; r1 may be m. This is the single-worker entry: it packs its own
// B panel. The multi-worker path (gemmBand) shares one packed B across workers.
func (g *gemmKernel[T]) packGemmRows(r0, r1 int, dst []T, a, b MatOf[T], k, n int, buf *packBuf[T]) {
	pa, pb := buf.pa, buf.pb
	for jc := 0; jc < n; jc += blockNC { // L3 column block of B
		nc := min(blockNC, n-jc)
		for pc := 0; pc < k; pc += blockKC { // L2 contraction block
			kc := min(blockKC, k-pc)
			g.packB(b, pc, kc, jc, nc, pb) // pack B(kc x nc) -> NR-wide panels
			g.gemmBand(r0, r1, jc, nc, pc, kc, dst, a, n, pa, pb)
		}
	}
}

// gemmBand runs the macro-kernel for the row band [r0,r1) against an
// already-packed B panel pb (the kc x nc block at source columns [jc,jc+nc),
// source contraction rows [pc,pc+kc)). It packs each MC-row block of A into pa
// (the worker's private buffer) and streams the register-blocked micro-kernel
// over the MR x NR tiles, accumulating into the band's rows of dst. Splitting B
// packing (done once by the caller) from this per-band A packing + macro-kernel
// is what lets several workers share one packed B: the redundant full-B repack
// each worker used to do — 1 MiB copied GOMAXPROCS times per kc block — was the
// dominant non-scaling cost of the old row-split parallelism.
func (g *gemmKernel[T]) gemmBand(r0, r1, jc, nc, pc, kc int, dst []T, a MatOf[T], n int, pa, pb []T) {
	mr0, nr0 := g.mr, g.nr
	tile := pa[len(pa)-mr0*nr0:] // see paCap
	if g.microUA != nil && unpackA && a.CS == 1 && a.RS > 0 {
		g.gemmBandUA(r0, r1, jc, nc, pc, kc, dst, a, n, pb, tile)
		return
	}
	for ic := r0; ic < r1; ic += blockMC { // L1 row block of A
		mc := min(blockMC, r1-ic)
		g.packA(a, ic, mc, pc, kc, pa) // pack A(mc x kc) -> MR-tall panels
		for jr := 0; jr < nc; jr += nr0 {
			nr := min(nr0, nc-jr)
			for ir := 0; ir < mc; ir += mr0 {
				mr := min(mr0, mc-ir)
				c0 := (ic+ir)*n + jc + jr
				if mr == mr0 && nr == nr0 {
					g.micro(kc, pa[ir*kc:], pb[jr*kc:], dst[c0:], n)
				} else {
					g.gemmEdge(kc, pa[ir*kc:], pb[jr*kc:], dst, c0, n, mr, nr, tile)
				}
			}
		}
	}
}

// gemmBandUA is gemmBand for a row-major A read in place by microUA: no A
// panel is packed. Each tile's A is the mr rows starting at row ic+ir, column
// pc, a.RS apart. A tile narrower than NR (the last B panel, zero-padded) runs
// into the scratch tile and only its nr columns are added into dst; a tile
// shorter than MR uses the kernel for its own height, so no row past A's last
// is read.
func (g *gemmKernel[T]) gemmBandUA(r0, r1, jc, nc, pc, kc int, dst []T, a MatOf[T], n int, pb, tile []T) {
	mr0, nr0 := g.mr, g.nr
	for jr := 0; jr < nc; jr += nr0 {
		nr := min(nr0, nc-jr)
		bp := pb[jr*kc:]
		for i := r0; i < r1; i += mr0 {
			mr := min(mr0, r1-i)
			ar := a.Data[a.Off+i*a.RS+pc:]
			c0 := i*n + jc + jr
			if nr == nr0 {
				g.microUA(kc, ar, a.RS, bp, dst[c0:], n, mr)
				continue
			}
			clear(tile)
			g.microUA(kc, ar, a.RS, bp, tile, nr0, mr)
			for r := 0; r < mr; r++ {
				row := dst[c0+r*n : c0+r*n+nr]
				for c := range row {
					row[c] += tile[r*nr0+c]
				}
			}
		}
	}
}

// packB copies B[pc:pc+kc][jc:jc+nc] into NR-wide column panels: for each panel
// jr the layout is pb[jr*kc + p*NR + c] = B(pc+p, jc+jr+c), with the c >= nr
// columns of an edge panel zero-filled so the micro-kernel reads a full NR.
// A unit column stride copies row runs; any other (a transposed B) walks each
// column down its own unit-or-not row stride, writing the panel column-wise.
func (g *gemmKernel[T]) packB(b MatOf[T], pc, kc, jc, nc int, pb []T) {
	NR := g.nr
	for jr := 0; jr < nc; jr += NR {
		nr := min(NR, nc-jr)
		dst := pb[jr*kc:]
		if b.CS == 1 {
			if nr == NR && packRowsWhole(dst, b.Data, b.Off+pc*b.RS+jc+jr, b.RS, kc, NR) {
				continue
			}
			// Element loops, not copy and clear: on runs this short (NR
			// elements) each of those is a runtime call.
			for p := 0; p < kc; p++ {
				srcRow := b.Data[b.Off+(pc+p)*b.RS+jc+jr:]
				d := dst[p*NR:]
				for c := 0; c < nr; c++ {
					d[c] = srcRow[c]
				}
				for c := nr; c < NR; c++ {
					d[c] = 0
				}
			}
			continue
		}
		for c := 0; c < nr; c++ {
			src := b.Off + pc*b.RS + (jc+jr+c)*b.CS
			for p := 0; p < kc; p++ {
				dst[p*NR+c] = b.Data[src+p*b.RS]
			}
		}
		for p := 0; p < kc; p++ {
			d := dst[p*NR:]
			for c := nr; c < NR; c++ {
				d[c] = 0
			}
		}
	}
}

// packRowsWhole copies kc runs of nr elements, the p-th starting at
// src[s+p*rs], to dst[p*nr:], each as one fixed-size array assignment, which
// the compiler lowers to a few vector moves instead of a loop or a runtime
// call. It reports false, copying nothing, for a width it has no array type
// for; the caller then copies element by element.
func packRowsWhole[T gemmFloat](dst, src []T, s, rs, kc, nr int) bool {
	switch nr {
	case 8:
		for p := 0; p < kc; p++ {
			*(*[8]T)(dst[p*8:]) = *(*[8]T)(src[s+p*rs:])
		}
	case 16:
		for p := 0; p < kc; p++ {
			*(*[16]T)(dst[p*16:]) = *(*[16]T)(src[s+p*rs:])
		}
	default:
		return false
	}
	return true
}

// packA copies A[ic:ic+mc][pc:pc+kc] into MR-tall row panels: for each panel ir
// the layout is pa[ir*kc + p*MR + r] = A(ic+ir+r, pc+p), with the r >= mr rows
// of an edge panel zero-filled so the micro-kernel reads a full MR.
func (g *gemmKernel[T]) packA(a MatOf[T], ic, mc, pc, kc int, pa []T) {
	MR := g.mr
	for ir := 0; ir < mc; ir += MR {
		mr := min(MR, mc-ir)
		dst := pa[ir*kc:]
		src := a.Off + (ic+ir)*a.RS + pc*a.CS
		if a.CS == 1 {
			// Row-major A: read each row as one contiguous run and scatter it
			// into its lane of the panel. The column-wise walk below spends
			// its loop and bounds checks on every MR elements; at 64² it
			// took longer than the micro-kernel.
			for r := 0; r < mr; r++ {
				row := a.Data[src+r*a.RS : src+r*a.RS+kc]
				d := dst[r : r+(kc-1)*MR+1]
				for p, v := range row {
					d[p*MR] = v
				}
			}
			for r := mr; r < MR; r++ {
				for p := 0; p < kc; p++ {
					dst[p*MR+r] = 0
				}
			}
			continue
		}
		for p := 0; p < kc; p++ {
			d := dst[p*MR:]
			at := src + p*a.CS
			for r := 0; r < mr; r++ {
				d[r] = a.Data[at+r*a.RS]
			}
			for r := mr; r < MR; r++ {
				d[r] = 0
			}
		}
	}
}

// gemmEdge computes a partial (mr < MR or nr < NR) tile with the micro-kernel
// itself, the way BLIS does: the packed panels are already zero-padded to a full
// MR x NR, so the kernel runs into a zeroed MR x NR scratch tile and only the
// valid mr x nr corner is added into dst. Every element of C is therefore summed
// by the same kernel, edge or not. The scalar loop this replaces ran ~20x slower
// per element, and on amd64 (MR=6) every 256-row MC block ends in a 4-row edge
// panel: it measured 40% of a 512^2 product there.
func (g *gemmKernel[T]) gemmEdge(kc int, pa, pb, dst []T, c0, ldc, mr, nr int, tile []T) {
	NR := g.nr
	clear(tile)
	g.micro(kc, pa, pb, tile, NR)
	for r := 0; r < mr; r++ {
		row := dst[c0+r*ldc : c0+r*ldc+nr]
		for c := range row {
			row[c] += tile[r*NR+c]
		}
	}
}

// GemmThreshold is the minimum number of result elements (m*n) at which the
// packed GEMM fans out across goroutines. Small products run single-threaded
// (one worker, one pooled buffer) so the goroutine launch never dominates.
//
// 6000 is the measured crossover on the Apple-silicon bench host: below it the
// goroutine launch + join costs more than the parallel speedup buys, above it
// the fan-out wins. Concretely, 64x64 (4096 results) is ~14us serial vs ~17us
// parallel (serial wins) while 80x80 (6400 results) is ~26us serial vs ~22us
// parallel and 96x96 (9216) is ~46us serial vs ~34us parallel (parallel wins by
// 15-30%). The old 1<<14 (16384) floor left the whole 80x80..120x120 band — the
// shapes that benefit most from the cores — stuck on the slow serial path; it
// was sized for MatVecP (memory-bound, see matVecThreshold), not for the
// compute-bound GEMM. 32x32 (1024) and 64x64 (4096) stay serial, where the
// register-blocked single-core kernel is fastest.
var GemmThreshold = 6000

// matVecThreshold is the m*k element count above which MatVecP fans the row-dots
// across cores. Mat-vec is a single bandwidth-bound streaming pass over A (one
// multiply-add per element, no reuse), so its parallel crossover is far higher
// than the compute-bound GEMM's: forking it on a few-thousand-element A is a net
// loss (100x100 mat-vec measures ~2.5us serial vs ~5.8us parallel). It is kept
// at the original 1<<14 floor — lowering GemmThreshold for the GEMM must not
// drag mat-vec onto the parallel path in the band where it regresses. It is a
// var so tests can pin it.
var matVecThreshold = 1 << 14

// MatMulP computes dst = a(m x k) * b(k x n) with the packed, cache-blocked GEMM,
// fanning the m output rows across cores above GemmThreshold. dst must be zeroed
// by the caller.
//
// Parallelism is the BLIS "loop-3" (ic, row-block) split with DYNAMIC, work-
// stealing scheduling: for every (jc, pc) cache block the B panel is packed ONCE
// into a shared buffer, then the m rows are diced into MR-aligned bands of MC
// rows and the workers pull bands off a shared atomic counter until exhausted,
// each packing only its own A panel and running the macro-kernel (gemmBand) over
// the band against the single shared packed B. Every band writes a disjoint,
// MR-aligned region of dst (b and the shared pb are read-only during the band),
// so the result is identical to the serial computation.
//
// Two properties together gave the large-N scaling:
//   - Packing B once (not once per worker) removes the O(P) redundant 1-MiB
//     repack the old row-split paid per kc block.
//   - Dynamic banding handles the M-series big.LITTLE asymmetry. A static
//     one-band-per-core split makes every fast P-core wait on the slow E-cores
//     at the join (measured: 16 cores was SLOWER than 10 on 1024², the 4 E-cores
//     straggling). With many small bands the P-cores simply grab more of them, so
//     adding the E-cores helps instead of hurting.
func MatMulP(dst, a, b []float64, m, k, n int) {
	MatMulStridedP(dst, rowMajor(a, k), rowMajor(b, n), m, k, n)
}

// MatMulStridedP is MatMulP over strided operand views (see Mat): dst (m x n,
// contiguous, zeroed by the caller) = a (m x k) * b (k x n). The views are
// only read, through the packing, so a transposed or sliced operand costs no
// copy beyond the one the packed GEMM makes anyway.
func MatMulStridedP(dst []float64, a, b Mat, m, k, n int) {
	gemm64.matMul(dst, a, b, m, k, n)
}

// HasGemm32 reports whether this architecture and CPU have a float32 GEMM
// micro-kernel, which MatMul32StridedP requires. Without one, the caller
// multiplies in float64.
func HasGemm32() bool { return gemm32 != nil }

// MatMul32StridedP is MatMulStridedP in float32, accumulating in float32 as a
// BLAS sgemm does. It requires HasGemm32.
func MatMul32StridedP(dst []float32, a, b Mat32, m, k, n int) {
	gemm32.matMul(dst, a, b, m, k, n)
}

func (g *gemmKernel[T]) matMul(dst []T, a, b MatOf[T], m, k, n int) {
	if m*n < GemmThreshold {
		buf := g.getPackBuf()
		g.packGemmRows(0, m, dst, a, b, k, n, buf)
		g.putPackBuf(buf)
		return
	}
	MR, NR := g.mr, g.nr
	w := runtime.GOMAXPROCS(0)
	if w > m {
		w = m
	}
	// Dice the m rows into MR-aligned bands a few rows of micro-tiles tall, so
	// there are several bands per worker. That oversubscription is what lets the
	// dynamic counter balance the M-series big.LITTLE cores: fast P-cores grab
	// more bands while the slow E-cores grab fewer, instead of every P-core
	// waiting on an E-core at a static join. bandRows is bandTilesMR micro-rows
	// (a multiple of MR so no packed A panel straddles a band boundary) but at
	// least one, and clamped so there are at least ~4 bands per worker when the
	// row count allows. Bands stay <= MC so each A panel still fits L1.
	bandRows := bandTilesMR * MR
	if want := roundUp(max(1, (m+(4*w)-1)/(4*w)), MR); want < bandRows {
		bandRows = want // small m: shrink bands so every worker still gets one
	}
	if bandRows > blockMC {
		bandRows = blockMC
	}
	nBands := (m + bandRows - 1) / bandRows
	if w > nBands {
		w = nBands // never spawn more goroutines than there is work for
	}

	// The workers are started once per call and stay for every (jc, pc) block,
	// instead of being forked and joined per block. A fork/join wakes the
	// parked worker threads one after another; measured on a 16-vCPU Zen 3
	// guest it costs 26 µs at 2 workers, 165 µs at 8 and 330 µs at 16, against
	// ~0.8 ms of work per block of a 1024² product, and there were two per
	// block (pack B, then the bands): ~40% of the wall time.
	//
	// The workers synchronise on work done, never on workers arrived: a block's
	// bands start once all of its B panels are packed and all bands of the
	// previous block are done. A worker still waking up holds nobody back, so
	// the wake-up chain stays off the critical path; a barrier that waited for
	// every worker put it right back on it for small products (96³ on 8 POWER9
	// cores: 0.57x). Claimed work never waits, so the scheme cannot deadlock.
	//
	// pb is double-buffered, so block i can be packed while block i-1 is being
	// multiplied. Tiles of C are still accumulated in pc order, since block i's
	// bands wait for block i-1's.
	pbs := [2][]T{g.getPB(), g.getPB()}
	defer g.putPB(pbs[0])
	defer g.putPB(pbs[1])

	type block struct{ jc, nc, pc, kc, chunks int }
	const chunk = 4 // B panels packed per claim
	var blocks []block
	for jc := 0; jc < n; jc += blockNC { // L3 column block of B
		nc := min(blockNC, n-jc)
		panels := (nc + NR - 1) / NR
		for pc := 0; pc < k; pc += blockKC { // L2 contraction block
			blocks = append(blocks, block{jc, nc, pc, min(blockKC, k-pc), (panels + chunk - 1) / chunk})
		}
	}
	type progress struct {
		packNext, packDone, bandNext, bandDone atomic.Int64
		_                                      [32]byte // keep blocks' counters off one cache line
	}
	prog := make([]progress, len(blocks))
	bandsDone := func(i int) bool { return i < 0 || prog[i].bandDone.Load() == int64(nBands) }

	work := func() {
		pa := g.getPA()
		defer g.putPA(pa)
		for i, bl := range blocks {
			// pbs[i%2] is free: its last reader was block i-2, and to get here
			// this worker passed block i-1's wait, which requires every band of
			// block i-2 done.
			pr, pb := &prog[i], pbs[i%2]
			for {
				c := int(pr.packNext.Add(1)) - 1
				if c >= bl.chunks {
					break
				}
				j0, j1 := c*chunk*NR, min((c+1)*chunk*NR, bl.nc)
				g.packB(b, bl.pc, bl.kc, bl.jc+j0, j1-j0, pb[j0*bl.kc:])
				pr.packDone.Add(1)
			}
			spinUntil(func() bool { return pr.packDone.Load() == int64(bl.chunks) && bandsDone(i-1) })
			for {
				bi := int(pr.bandNext.Add(1)) - 1
				if bi >= nBands {
					break
				}
				r0 := bi * bandRows
				g.gemmBand(r0, min(r0+bandRows, m), bl.jc, bl.nc, bl.pc, bl.kc, dst, a, n, pa, pb)
				pr.bandDone.Add(1)
			}
		}
	}
	var wg sync.WaitGroup
	for range w - 1 {
		wg.Add(1)
		go func() { defer wg.Done(); work() }()
	}
	work() // the caller is worker 0
	wg.Wait()
}

// MatMul computes dst = a(m x k) * b(k x n) serially with the packed GEMM (the
// single-worker path of MatMulP). dst must be zeroed by the caller. It is kept as
// the named serial entry point used by callers and tests that want the exact
// non-parallel computation.
func MatMul(dst, a, b []float64, m, k, n int) {
	buf := gemm64.getPackBuf()
	gemm64.packGemmRows(0, m, dst, rowMajor(a, k), rowMajor(b, n), k, n, buf)
	gemm64.putPackBuf(buf)
}

// dotRange returns sum(a[i]*b[i]) over equal-length slices, using four
// independent accumulators, so four multiply-add chains are in flight instead of
// one (gc does not vectorise this loop; the gain is latency hiding). The
// four-way grouping is a fixed
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
// fanned across cores above matVecThreshold. Each output dst[i] is the dot of
// row i of a (the contiguous slice a[i*k:(i+1)*k]) with v, so the access pattern
// is pure unit-stride streaming — far cheaper than routing a single-column GEMM
// through the packer. dst need not be pre-zeroed (each entry is written, not
// accumulated). Rows are disjoint, so the parallel result equals the serial one.
func MatVecP(dst, a, v []float64, m, k int) { MatVecStridedP(dst, a, k, v, m, k) }

// MatVecStridedP is MatVecP on rows lda elements apart: row i of a is
// a[i*lda : i*lda+k], so a sub-block of a larger row-major matrix is
// multiplied in place.
func MatVecStridedP(dst, a []float64, lda int, v []float64, m, k int) {
	body := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = dotSIMD(a[i*lda:i*lda+k], v)
		}
	}
	if m*k < matVecThreshold {
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
// fanned across cores above ParThreshold: each chunk is dot-producted with the
// SIMD kernel and the partials are summed. The chunked four-way
// grouping reassociates the sum (a few ULP, like SumP / numpy pairwise); callers
// needing the exact left-to-right value use the scalar Dot1D.
func Dot1DP(a, b []float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return dotSIMD(a, b)
	}
	w := numWorkers(n)
	parts := min(4*w, max(w, n/reduceGrain)) // see mapReduceP
	partials := make([]float64, parts)
	chunk := (n + parts - 1) / parts
	parallelFor(parts, w, func(lo, hi int) {
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
			partials[idx] = dotSIMD(a[s:e], b[s:e])
		}
	})
	return Sum(partials)
}
