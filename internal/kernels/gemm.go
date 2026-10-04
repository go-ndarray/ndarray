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

// Mat is a read-only strided view of a matrix: element (i, j) is
// Data[Off + i*RS + j*CS]. Packing reads the operands through it, so a
// transposed, sliced or offset view is multiplied in place — the packing copy
// the GEMM makes anyway does the gather — instead of being materialised into
// a contiguous copy first (what BLAS's trans flags buy). Strides may be
// negative or zero; every (i, j) in range must index Data.
type Mat struct {
	Data        []float64
	Off, RS, CS int
}

// rowMajor views a contiguous row-major matrix with ld elements per row.
func rowMajor(data []float64, ld int) Mat { return Mat{Data: data, RS: ld, CS: 1} }

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
// band boundary; r1 may be m. This is the single-worker entry: it packs its own
// B panel. The multi-worker path (gemmBand) shares one packed B across workers.
func packGemmRows(r0, r1 int, dst []float64, a, b Mat, k, n int, buf *packBuf) {
	pa, pb := buf.pa, buf.pb
	for jc := 0; jc < n; jc += blockNC { // L3 column block of B
		nc := min(blockNC, n-jc)
		for pc := 0; pc < k; pc += blockKC { // L2 contraction block
			kc := min(blockKC, k-pc)
			packB(b, pc, kc, jc, nc, pb) // pack B(kc x nc) -> NR-wide panels
			gemmBand(r0, r1, jc, nc, pc, kc, dst, a, n, pa, pb)
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
func gemmBand(r0, r1, jc, nc, pc, kc int, dst []float64, a Mat, n int, pa, pb []float64) {
	for ic := r0; ic < r1; ic += blockMC { // L1 row block of A
		mc := min(blockMC, r1-ic)
		packA(a, ic, mc, pc, kc, pa) // pack A(mc x kc) -> MR-tall panels
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

// packB copies B[pc:pc+kc][jc:jc+nc] into NR-wide column panels: for each panel
// jr the layout is pb[jr*kc + p*NR + c] = B(pc+p, jc+jr+c), with the c >= nr
// columns of an edge panel zero-filled so the micro-kernel reads a full NR.
// A unit column stride copies row runs; any other (a transposed B) walks each
// column down its own unit-or-not row stride, writing the panel column-wise.
func packB(b Mat, pc, kc, jc, nc int, pb []float64) {
	for jr := 0; jr < nc; jr += NR {
		nr := min(NR, nc-jr)
		dst := pb[jr*kc:]
		if b.CS == 1 {
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

// packA copies A[ic:ic+mc][pc:pc+kc] into MR-tall row panels: for each panel ir
// the layout is pa[ir*kc + p*MR + r] = A(ic+ir+r, pc+p), with the r >= mr rows
// of an edge panel zero-filled so the micro-kernel reads a full MR.
func packA(a Mat, ic, mc, pc, kc int, pa []float64) {
	for ir := 0; ir < mc; ir += MR {
		mr := min(MR, mc-ir)
		dst := pa[ir*kc:]
		src := a.Off + (ic+ir)*a.RS + pc*a.CS
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
func gemmEdge(kc int, pa, pb, dst []float64, c0, ldc, mr, nr int) {
	var tile [MR * NR]float64
	gemmMicro(kc, pa, pb, tile[:], NR)
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
	// meeting at a spinning barrier between blocks instead of being forked and
	// joined per block. A fork/join wakes the parked worker threads one after
	// another; measured on a 16-vCPU Zen 3 guest it costs 26 µs at 2 workers,
	// 165 µs at 8 and 330 µs at 16, against ~0.8 ms of work per block of a
	// 1024² product. With two fork/joins per block (pack B, then the bands)
	// that was ~40% of the wall time, and why the CPUs were busy only 7 of 16.
	//
	// pb is double-buffered so one barrier per block suffices: block i packs
	// into pbs[i%2] while nobody can still be reading it, because reading it
	// for block i-2 ended before everyone passed block i-1's barrier.
	shared, shared2 := getPackBuf(), getPackBuf()
	defer packPool.Put(shared)
	defer packPool.Put(shared2)
	pbs := [2][]float64{shared.pb, shared2.pb}

	type block struct{ jc, nc, pc, kc int }
	var blocks []block
	for jc := 0; jc < n; jc += blockNC { // L3 column block of B
		for pc := 0; pc < k; pc += blockKC { // L2 contraction block
			blocks = append(blocks, block{jc, min(blockNC, n-jc), pc, min(blockKC, k-pc)})
		}
	}
	// Each block has its own band cursor (dynamic banding within a block) and
	// its own B-panel cursor (packing shared by whoever is free).
	bandNext := make([]atomic.Int64, len(blocks))
	packNext := make([]atomic.Int64, len(blocks))
	bar := newSpinBarrier(w)

	work := func() {
		ab := getPackBuf()
		defer packPool.Put(ab)
		for i, bl := range blocks {
			pb := pbs[i%2]
			panels := (bl.nc + NR - 1) / NR
			for {
				// B panels in chunks, so packing is shared without one atomic
				// per 8-column panel.
				const chunk = 4
				p0 := int(packNext[i].Add(chunk)) - chunk
				if p0 >= panels {
					break
				}
				j0, j1 := p0*NR, min((p0+chunk)*NR, bl.nc)
				packB(b, bl.pc, bl.kc, bl.jc+j0, j1-j0, pb[j0*bl.kc:])
			}
			bar.wait() // pb for block i is complete
			for {
				bi := int(bandNext[i].Add(1)) - 1
				if bi >= nBands {
					break
				}
				r0 := bi * bandRows
				gemmBand(r0, min(r0+bandRows, m), bl.jc, bl.nc, bl.pc, bl.kc, dst, a, n, ab.pa, pb)
			}
			// No barrier here: the next block packs into the other buffer, and
			// the barrier after that packing orders this block's reads of pb
			// before the packing of block i+2 into it. Tiles of C are still
			// accumulated in pc order: a band of block i+1 starts only after
			// every worker has passed block i+1's barrier, i.e. finished its
			// bands of block i.
		}
	}
	var wg sync.WaitGroup
	for g := 1; g < w; g++ {
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
	buf := getPackBuf()
	packGemmRows(0, m, dst, rowMajor(a, k), rowMajor(b, n), k, n, buf)
	packPool.Put(buf)
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
func MatVecP(dst, a, v []float64, m, k int) {
	body := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			dst[i] = dotSIMD(a[i*k:i*k+k], v)
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
// fanned across cores above ParThreshold: each worker dot-products its chunk with
// the unrolled dotRange and the partials are summed. The chunked four-way
// grouping reassociates the sum (a few ULP, like SumP / numpy pairwise); callers
// needing the exact left-to-right value use the scalar Dot1D.
func Dot1DP(a, b []float64) float64 {
	n := len(a)
	if n < ParThreshold {
		return dotSIMD(a, b)
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
			partials[idx] = dotSIMD(a[s:e], b[s:e])
		}
	})
	return Sum(partials)
}
