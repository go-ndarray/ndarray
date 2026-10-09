package random

import (
	"math"

	"github.com/go-ndarray/ndarray"
)

// intBounds are the bounds of numpy's integer dtypes, as int64 (uint64's
// upper bound does not fit and is handled by IntegersUint64).
var intBounds = map[ndarray.DType][2]int64{
	ndarray.Bool:   {0, 1},
	ndarray.Int8:   {math.MinInt8, math.MaxInt8},
	ndarray.Int16:  {math.MinInt16, math.MaxInt16},
	ndarray.Int32:  {math.MinInt32, math.MaxInt32},
	ndarray.Int64:  {math.MinInt64, math.MaxInt64},
	ndarray.Uint8:  {0, math.MaxUint8},
	ndarray.Uint16: {0, math.MaxUint16},
	ndarray.Uint32: {0, math.MaxUint32},
	ndarray.Uint64: {0, math.MaxInt64},
}

// boundsErr is numpy's format_bounds_error.
func boundsErr(closed bool, low int64) error {
	switch {
	case low == 0 && closed:
		return valueErr("high < 0")
	case low == 0:
		return valueErr("high <= 0")
	case closed:
		return valueErr("low > high")
	}
	return valueErr("low >= high")
}

// Integers returns int64 values in [low, high), numpy's
// integers(low, high, size).
func (g *Generator) Integers(low, high int64, size ...int) (*ndarray.Array, error) {
	return g.IntegersOf(ndarray.Int64, low, high, false, size...)
}

// IntegersOf is numpy's integers(low, high, size, dtype, endpoint): values in
// [low, high), or [low, high] with endpoint, of an integer or Bool dtype,
// drawn with Lemire's method.
func (g *Generator) IntegersOf(dt ndarray.DType, low, high int64, endpoint bool, size ...int) (*ndarray.Array, error) {
	return integers(g.bg, false, dt, low, high, endpoint, size)
}

// IntegersUint64 is numpy's integers(low, high, size, dtype=np.uint64,
// endpoint) for bounds beyond int64.
func (g *Generator) IntegersUint64(low, high uint64, endpoint bool, size ...int) (*ndarray.Array, error) {
	return integersUint64(g.bg, false, low, high, endpoint, size)
}

func integersUint64(bg BitGenerator, masked bool, low, high uint64, endpoint bool, size []int) (*ndarray.Array, error) {
	if !endpoint {
		if high == 0 {
			return nil, boundsErr(false, int64(min(low, 1)))
		}
		high--
	}
	if low > high {
		return nil, boundsErr(true, int64(min(low, 1)))
	}
	a, d, err := newArray[uint64](ndarray.Uint64, size)
	if err != nil {
		return nil, err
	}
	fillUint64(bg, low, high-low, masked, d)
	return a, nil
}

func integers(bg BitGenerator, masked bool, dt ndarray.DType, low, high int64, endpoint bool, size []int) (*ndarray.Array, error) {
	lim, ok := intBounds[dt]
	if !ok {
		return nil, valueErr("Unsupported dtype %v for integers", dt)
	}
	if !endpoint {
		if high == math.MinInt64 {
			return nil, boundsErr(false, low)
		}
		high--
	}
	if low < lim[0] {
		return nil, valueErr("low is out of bounds for %v", dt)
	}
	if high > lim[1] {
		return nil, valueErr("high is out of bounds for %v", dt)
	}
	if low > high {
		return nil, boundsErr(endpoint, low)
	}
	if dt == ndarray.Uint64 {
		return integersUint64(bg, masked, uint64(low), uint64(high), true, size)
	}
	rng := uint64(high - low)
	off := uint64(low)
	a, err := ndarray.ZerosOf(dt, size...)
	if err != nil {
		return nil, err
	}
	switch dt {
	case ndarray.Bool:
		d, _ := ndarray.Data[bool](a)
		fillBool(bg, off != 0, rng != 0, d)
	case ndarray.Int8:
		d, _ := ndarray.Data[int8](a)
		fillUint8(bg, uint8(off), uint8(rng), masked, asUint8(d))
	case ndarray.Uint8:
		d, _ := ndarray.Data[uint8](a)
		fillUint8(bg, uint8(off), uint8(rng), masked, d)
	case ndarray.Int16:
		d, _ := ndarray.Data[int16](a)
		fillUint16(bg, uint16(off), uint16(rng), masked, asUint16(d))
	case ndarray.Uint16:
		d, _ := ndarray.Data[uint16](a)
		fillUint16(bg, uint16(off), uint16(rng), masked, d)
	case ndarray.Int32:
		d, _ := ndarray.Data[int32](a)
		fillUint32(bg, uint32(off), uint32(rng), masked, asUint32(d))
	case ndarray.Uint32:
		d, _ := ndarray.Data[uint32](a)
		fillUint32(bg, uint32(off), uint32(rng), masked, d)
	default: // Int64
		d, _ := ndarray.Data[int64](a)
		fillUint64(bg, off, rng, masked, asUint64(d))
	}
	return a, nil
}
