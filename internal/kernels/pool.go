package kernels

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// A persistent pool of helper goroutines for parallelFor.
//
// Forking goroutines per call wakes parked threads one after another, and a
// call waited for the slowest wake-up: on a POWER9, a 2^20-element dot ran
// slower on 8 cores (553 µs) than on 4 (212 µs), with 2.4 of 8 CPUs busy. The
// helpers here are started once and, after a job, keep polling for the next one
// for spinWindow before they park, as OpenBLAS's and OpenMP's threads do, so
// operations issued back to back find them awake. A parked helper is woken when
// a job is published; until it arrives, the others, the caller included, take
// its blocks.

// spinWindow is how long, in nanoseconds, an idle helper keeps polling for the
// next job before it parks. Atomic because the helpers read it while tests set it.
var spinWindow atomic.Int64

func init() { spinWindow.Store(int64(200 * time.Microsecond)) }

type poolJob struct {
	run func(g int) // g in [1, w): the helper's owner index
	w   int
}

type poolHelper struct {
	parked atomic.Bool
	wake   chan struct{}
}

var pool struct {
	busy    sync.Mutex // one job at a time; a concurrent caller forks instead
	grow    sync.Mutex
	helpers []*poolHelper
	gen     atomic.Uint64
	job     atomic.Pointer[poolJob]
}

// ensureHelpers starts helpers until there are at least k.
func ensureHelpers(k int) {
	pool.grow.Lock()
	defer pool.grow.Unlock()
	for len(pool.helpers) < k {
		h := &poolHelper{wake: make(chan struct{}, 1)}
		i := len(pool.helpers)
		pool.helpers = append(pool.helpers, h)
		go helperLoop(i, h, pool.gen.Load())
	}
}

// helperLoop runs jobs as helper i: owner index i+1 of each job that has room
// for it.
func helperLoop(i int, h *poolHelper, seen uint64) {
	for {
		start := time.Now()
		for spins := 0; pool.gen.Load() == seen; spins++ {
			if spins&255 != 255 {
				continue
			}
			if time.Since(start) < time.Duration(spinWindow.Load()) {
				runtime.Gosched()
				continue
			}
			// Park. Re-check after announcing it: a job published in
			// between either saw the flag and sends a wake-up (wait for it,
			// so no stale one is left), or this helper takes the flag back
			// and goes straight to the job.
			h.parked.Store(true)
			if pool.gen.Load() == seen || !h.parked.CompareAndSwap(true, false) {
				<-h.wake
			}
			break
		}
		seen = pool.gen.Load()
		if j := pool.job.Load(); i+1 < j.w { // a job is published before gen moves
			j.run(i + 1)
		}
	}
}

// noJob replaces a finished job in pool.job. Left there, the job's closure would
// keep the operation's arrays alive until the next parallel operation. Its w of
// 0 admits no helper.
var noJob = &poolJob{}

// retire drops the reference to the finished job. A helper that loaded it just
// before finds every block taken and lets go of it on return.
func retire() { pool.job.Store(noJob) }

// publish hands run to the first w-1 helpers and wakes the parked ones.
func publish(run func(g int), w int) {
	pool.job.Store(&poolJob{run: run, w: w})
	pool.gen.Add(1)
	for _, h := range pool.helpers[:w-1] {
		if h.parked.CompareAndSwap(true, false) {
			h.wake <- struct{}{}
		}
	}
}
