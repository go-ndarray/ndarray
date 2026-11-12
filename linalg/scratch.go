package linalg

import "sync"

// scratchMax is the largest buffer (in float64s, 8 MiB) the scratch pool keeps:
// a bigger one is allocated and dropped, so a single huge product does not pin
// its temporary for the life of the process.
const scratchMax = 1 << 20

var scratchPool sync.Pool // of *[]float64

// getScratch returns a zeroed float64 buffer of length n, reused across calls
// when one is free: the blocked factorizations call fastGemm once per panel,
// and its temporary was one allocation each time (#104).
func getScratch(n int) *[]float64 {
	if p, ok := scratchPool.Get().(*[]float64); ok && cap(*p) >= n {
		*p = (*p)[:n]
		clear(*p)
		return p
	}
	s := make([]float64, n)
	return &s
}

// putScratch returns p to the pool unless it is over scratchMax.
func putScratch(p *[]float64) {
	if cap(*p) <= scratchMax {
		scratchPool.Put(p)
	}
}
