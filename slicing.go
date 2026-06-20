package ndarray

import (
	"errors"
	"fmt"
)

// ErrIndex is returned when an integer index or slice argument is out of range
// for the axis it addresses, or when the number of index arguments does not
// match the array's rank.
var ErrIndex = errors.New("ndarray: index out of range")

// Index describes how a single axis is indexed by Slice. It is either an
// integer index (which removes that axis from the result, NumPy basic-indexing
// semantics) or a half-open range start:stop:step (which keeps the axis as a
// strided view). Construct one with A (integer) or R / Rng (range).
type Index struct {
	// isInt selects integer-index mode; otherwise this is a range.
	isInt bool
	// i is the integer index when isInt is true.
	i int
	// start, stop, step describe the range when isInt is false. hasStart and
	// hasStop record whether the bound was given explicitly, so it can default
	// to the natural extreme for the sign of step (NumPy slice semantics).
	start, stop, step int
	hasStart, hasStop bool
}

// A returns an integer Index that selects position i along an axis and removes
// that axis from the result, mirroring NumPy's a[i] basic indexing. A negative
// i counts from the end of the axis.
func A(i int) Index { return Index{isInt: true, i: i} }

// All returns a range Index covering an entire axis with step 1, the
// equivalent of NumPy's a[:].
func All() Index { return Index{step: 1} }

// R returns a range Index for start:stop with step 1, keeping the axis as a
// view (NumPy a[start:stop]). Negative bounds count from the end of the axis.
func R(start, stop int) Index {
	return Index{start: start, stop: stop, step: 1, hasStart: true, hasStop: true}
}

// Rng returns a range Index for start:stop:step, keeping the axis as a view
// (NumPy a[start:stop:step]). Negative bounds count from the end; step may be
// negative to reverse, but must not be zero.
func Rng(start, stop, step int) Index {
	return Index{start: start, stop: stop, step: step, hasStart: true, hasStop: true}
}

// From returns a range Index for start: (to the end of the axis) with step 1
// (NumPy a[start:]).
func From(start int) Index {
	return Index{start: start, step: 1, hasStart: true}
}

// To returns a range Index for :stop (from the start of the axis) with step 1
// (NumPy a[:stop]).
func To(stop int) Index {
	return Index{stop: stop, step: 1, hasStop: true}
}

// Step returns a range Index covering the whole axis with the given step
// (NumPy a[::step]); step may be negative to reverse the axis, but not zero.
func Step(step int) Index {
	return Index{step: step}
}

// clampIndex normalises an integer index for an axis of the given length,
// applying NumPy's negative-from-end rule and bounds checking.
func clampIndex(i, length int) (int, error) {
	if i < 0 {
		i += length
	}
	if i < 0 || i >= length {
		return 0, fmt.Errorf("%w: index %d for axis of length %d", ErrIndex, i, length)
	}
	return i, nil
}

// resolveRange computes the (offset-element start, count, step) of a range
// Index against an axis of the given length, following NumPy slice semantics:
// negative bounds count from the end, unset bounds default to the natural
// extreme for the sign of step, bounds are clamped into range, and the element
// count is ceil((stop-start)/step) clamped to >= 0.
func (ix Index) resolveRange(length int) (first, count, step int, err error) {
	step = ix.step
	if step == 0 {
		return 0, 0, 0, fmt.Errorf("%w: slice step must be non-zero", ErrIndex)
	}

	// Defaults depend on the direction of the step.
	if step > 0 {
		first = 0
		if ix.hasStart {
			first = normBound(ix.start, length, 0, length)
		}
		stop := length
		if ix.hasStop {
			stop = normBound(ix.stop, length, 0, length)
		}
		count = 0
		if stop > first {
			count = (stop - first + step - 1) / step
		}
		return first, count, step, nil
	}

	// step < 0: iterate downward; defaults are last element down to before -1.
	first = length - 1
	if ix.hasStart {
		first = normBound(ix.start, length, -1, length-1)
	}
	stop := -1
	if ix.hasStop {
		stop = normBound(ix.stop, length, -1, length-1)
	}
	count = 0
	if first > stop {
		count = (first - stop - step - 1) / (-step)
	}
	return first, count, step, nil
}

// normBound applies NumPy's slice-bound normalisation: a negative bound counts
// from the end, then the result is clamped to [lo, hi].
func normBound(b, length, lo, hi int) int {
	if b < 0 {
		b += length
	}
	if b < lo {
		b = lo
	}
	if b > hi {
		b = hi
	}
	return b
}

// Slice returns a view of the array selected by per-axis indices, following
// NumPy basic-indexing semantics. It accepts exactly Ndim index arguments
// (build them with A, All, R, Rng, From, To). Integer indices (A) drop their
// axis; range indices keep the axis as a strided view that shares the receiver's
// backing data — writes through the view are visible in the original and vice
// versa. The result may have a non-zero offset and arbitrary strides.
func (a *Array) Slice(idx ...Index) (*Array, error) {
	if len(idx) != len(a.shape) {
		return nil, fmt.Errorf("%w: got %d indices for %d-dimensional array",
			ErrIndex, len(idx), len(a.shape))
	}
	offset := a.offset
	shape := make([]int, 0, len(a.shape))
	strides := make([]int, 0, len(a.shape))
	for axis, ix := range idx {
		length := a.shape[axis]
		if ix.isInt {
			i, err := clampIndex(ix.i, length)
			if err != nil {
				return nil, fmt.Errorf("axis %d: %w", axis, err)
			}
			offset += i * a.strides[axis]
			continue // integer index removes the axis
		}
		first, count, step, err := ix.resolveRange(length)
		if err != nil {
			return nil, fmt.Errorf("axis %d: %w", axis, err)
		}
		offset += first * a.strides[axis]
		shape = append(shape, count)
		strides = append(strides, step*a.strides[axis])
	}
	return &Array{data: a.data, shape: shape, strides: strides, offset: offset}, nil
}
