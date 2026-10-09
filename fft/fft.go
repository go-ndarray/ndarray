// Package fft is numpy.fft for ndarray arrays: the discrete Fourier
// transforms of go-fft (github.com/go-fft/fft), applied along any axes of an
// N-dimensional array, with numpy's n/s, axes and norm arguments and its
// dtypes.
//
// Every transform takes an array of any dtype. As in NumPy 2, float32 and
// complex64 inputs are transformed in single precision and give complex64
// (or float32 for the real inverses); every other dtype is transformed in
// double precision. Problems NumPy reports with an exception — an axis out of
// range or repeated, a length below one — are errors here.
package fft

import (
	"errors"
	"fmt"

	gofft "github.com/go-fft/fft"
	"github.com/go-ndarray/ndarray"
)

// Norm selects how a forward/inverse pair is scaled, as numpy.fft's norm
// argument does. The zero value, Backward, is numpy's default: the forward
// transform is unscaled and the inverse divides by n.
type Norm = gofft.Norm

// The three scalings.
const (
	Backward = gofft.NormBackward
	Ortho    = gofft.NormOrtho
	Forward  = gofft.NormForward
)

// ErrArgs is returned for arguments numpy.fft rejects: an axis out of range,
// a length below one, S and Axes of different lengths, or a complex input to
// a transform of real signals. An axis listed twice is accepted, as NumPy
// accepts it: the transform runs along it twice.
var ErrArgs = errors.New("fft: invalid arguments")

// Options carries numpy.fft's optional arguments. Its zero value is numpy's
// defaults, so FFTWith(a, Options{}) is FFT(a).
type Options struct {
	// S is numpy's n (for the 1-D transforms, one entry) or s (for the
	// others): the length of each transformed axis. The input is truncated or
	// zero-padded to it along that axis. For the real inverses (IRFFT,
	// IRFFTN, HFFT) it is the length of the real output along the last
	// transformed axis. nil keeps the input's lengths (and, for the real
	// inverses, 2*(m-1) along the last axis, m being its number of bins).
	S []int

	// Axes are the axes to transform; negative values count from the end.
	// nil means numpy's default: the last axis for the 1-D transforms, the
	// last two for the 2-D ones, all of them for the N-D ones (or the last
	// len(S) when S is set).
	Axes []int

	// Norm scales the result.
	Norm Norm
}

// kind is the family of a transform.
type kind uint8

const (
	c2c   kind = iota // complex to complex
	r2c               // real to half complex
	c2r               // half complex to real
	herm              // hfft: half Hermitian to real, scaled as a forward transform
	iherm             // ihfft: real to half Hermitian, scaled as an inverse transform
)

// FFT is numpy.fft.fft: the 1-D discrete Fourier transform along the last
// axis.
func FFT(a *ndarray.Array) (*ndarray.Array, error) { return FFTWith(a, Options{}) }

// FFTWith is numpy.fft.fft(a, n, axis, norm).
func FFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, false, oneD)
}

// IFFT is numpy.fft.ifft: the inverse of FFT.
func IFFT(a *ndarray.Array) (*ndarray.Array, error) { return IFFTWith(a, Options{}) }

// IFFTWith is numpy.fft.ifft(a, n, axis, norm).
func IFFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, true, oneD)
}

// RFFT is numpy.fft.rfft: the transform of a real signal along the last
// axis, keeping the n/2+1 non-negative frequencies. A complex input is an
// error, as in NumPy 2.
func RFFT(a *ndarray.Array) (*ndarray.Array, error) { return RFFTWith(a, Options{}) }

// RFFTWith is numpy.fft.rfft(a, n, axis, norm).
func RFFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, r2c, false, oneD)
}

// IRFFT is numpy.fft.irfft: the inverse of RFFT, a real signal of length
// 2*(m-1) for m bins unless Options.S says otherwise.
func IRFFT(a *ndarray.Array) (*ndarray.Array, error) { return IRFFTWith(a, Options{}) }

// IRFFTWith is numpy.fft.irfft(a, n, axis, norm).
func IRFFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2r, true, oneD)
}

// HFFT is numpy.fft.hfft: the transform of a signal with Hermitian symmetry,
// given by its first half, which is real.
func HFFT(a *ndarray.Array) (*ndarray.Array, error) { return HFFTWith(a, Options{}) }

// HFFTWith is numpy.fft.hfft(a, n, axis, norm).
func HFFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, herm, false, oneD)
}

// IHFFT is numpy.fft.ihfft: the inverse of HFFT.
func IHFFT(a *ndarray.Array) (*ndarray.Array, error) { return IHFFTWith(a, Options{}) }

// IHFFTWith is numpy.fft.ihfft(a, n, axis, norm).
func IHFFTWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, iherm, true, oneD)
}

// FFT2 is numpy.fft.fft2: the transform over the last two axes.
func FFT2(a *ndarray.Array) (*ndarray.Array, error) { return FFT2With(a, Options{}) }

// FFT2With is numpy.fft.fft2(a, s, axes, norm).
func FFT2With(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, false, twoD)
}

// IFFT2 is numpy.fft.ifft2.
func IFFT2(a *ndarray.Array) (*ndarray.Array, error) { return IFFT2With(a, Options{}) }

// IFFT2With is numpy.fft.ifft2(a, s, axes, norm).
func IFFT2With(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, true, twoD)
}

// RFFT2 is numpy.fft.rfft2.
func RFFT2(a *ndarray.Array) (*ndarray.Array, error) { return RFFT2With(a, Options{}) }

// RFFT2With is numpy.fft.rfft2(a, s, axes, norm).
func RFFT2With(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, r2c, false, twoD)
}

// IRFFT2 is numpy.fft.irfft2.
func IRFFT2(a *ndarray.Array) (*ndarray.Array, error) { return IRFFT2With(a, Options{}) }

// IRFFT2With is numpy.fft.irfft2(a, s, axes, norm).
func IRFFT2With(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2r, true, twoD)
}

// FFTN is numpy.fft.fftn: the transform over every axis.
func FFTN(a *ndarray.Array) (*ndarray.Array, error) { return FFTNWith(a, Options{}) }

// FFTNWith is numpy.fft.fftn(a, s, axes, norm).
func FFTNWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, false, allD)
}

// IFFTN is numpy.fft.ifftn.
func IFFTN(a *ndarray.Array) (*ndarray.Array, error) { return IFFTNWith(a, Options{}) }

// IFFTNWith is numpy.fft.ifftn(a, s, axes, norm).
func IFFTNWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2c, true, allD)
}

// RFFTN is numpy.fft.rfftn: the real transform runs along the last axis.
func RFFTN(a *ndarray.Array) (*ndarray.Array, error) { return RFFTNWith(a, Options{}) }

// RFFTNWith is numpy.fft.rfftn(a, s, axes, norm).
func RFFTNWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, r2c, false, allD)
}

// IRFFTN is numpy.fft.irfftn.
func IRFFTN(a *ndarray.Array) (*ndarray.Array, error) { return IRFFTNWith(a, Options{}) }

// IRFFTNWith is numpy.fft.irfftn(a, s, axes, norm).
func IRFFTNWith(a *ndarray.Array, o Options) (*ndarray.Array, error) {
	return run(a, o, c2r, true, allD)
}

// rank is how many axes a transform family takes by default.
type rank uint8

const (
	oneD rank = iota
	twoD
	allD
)

// run resolves the arguments, pads or truncates the input, and calls go-fft.
func run(a *ndarray.Array, o Options, k kind, inverse bool, r rank) (*ndarray.Array, error) {
	if (k == r2c || k == iherm) && a.DType().IsComplex() {
		return nil, fmt.Errorf("%w: a real transform of a %v array", ErrArgs, a.DType())
	}
	axes, dup, err := resolveAxes(a, o, r)
	if err != nil {
		return nil, err
	}
	out, err := outLengths(a, o, k, axes)
	if err != nil {
		return nil, err
	}
	if dup {
		return sequential(a, axes, out, o.Norm, k, inverse)
	}
	// The input is cut or padded to the lengths the transform reads: for the
	// half-spectrum inverses the last axis holds out/2+1 bins.
	in := append([]int(nil), out...)
	if k == c2r || k == herm {
		in[len(in)-1] = out[len(out)-1]/2 + 1
	}
	x := a
	for i, ax := range axes {
		x = resize(x, ax, in[i])
	}
	single := a.DType() == ndarray.Float32 || a.DType() == ndarray.Complex64
	shape := x.Shape()
	switch k {
	case c2c:
		return complexTransform(x, shape, axes, o.Norm, inverse, single), nil
	case r2c:
		return realForward(x, shape, axes, o.Norm, single), nil
	case c2r:
		shape[axes[len(axes)-1]] = out[len(out)-1]
		return realInverse(x, shape, axes, o.Norm, single), nil
	case herm:
		// numpy: hfft(a, n) = irfft(conj(a), n) scaled as a forward transform.
		shape[axes[len(axes)-1]] = out[len(out)-1]
		return realInverse(x.Conj(), shape, axes, opposite(o.Norm), single), nil
	}
	// iherm: ihfft(a, n) = conj(rfft(a, n)) scaled as an inverse transform.
	return realForward(x, shape, axes, opposite(o.Norm), single).Conj(), nil
}

// sequential runs a multi-axis transform one axis at a time, in the order
// numpy.fft does, for an axis list that names an axis twice: that axis is then
// transformed twice, each time cut or padded to its own length. (go-fft
// refuses a repeated axis.) fftn goes from the last axis to the first; rfftn
// does the real transform along the last axis first, then the others; irfftn
// does the others first and the real inverse last.
func sequential(a *ndarray.Array, axes, out []int, norm Norm, k kind, inverse bool) (*ndarray.Array, error) {
	one := func(x *ndarray.Array, i int, k kind) (*ndarray.Array, error) {
		return run(x, Options{S: []int{out[i]}, Axes: []int{axes[i]}, Norm: norm}, k, inverse, oneD)
	}
	last := len(axes) - 1
	x := a
	var err error
	switch k {
	case c2c:
		for i := last; i >= 0 && err == nil; i-- {
			x, err = one(x, i, c2c)
		}
	case r2c:
		x, err = one(x, last, r2c)
		for i := 0; i < last && err == nil; i++ {
			x, err = one(x, i, c2c)
		}
	default: // c2r: herm and iherm are 1-D and cannot repeat an axis
		for i := 0; i < last && err == nil; i++ {
			x, err = one(x, i, c2c)
		}
		if err == nil {
			x, err = one(x, last, c2r)
		}
	}
	return x, err
}

// opposite swaps the forward and inverse scalings.
func opposite(n Norm) Norm {
	switch n {
	case Backward:
		return Forward
	case Forward:
		return Backward
	}
	return n
}

// resolveAxes returns the normalised axes to transform, and whether one is
// listed twice.
func resolveAxes(a *ndarray.Array, o Options, r rank) (axes []int, dup bool, err error) {
	nd := a.Ndim()
	axes = o.Axes
	if axes == nil {
		switch {
		case r == oneD:
			axes = []int{-1}
		case r == twoD:
			axes = []int{-2, -1}
		case o.S != nil:
			axes = make([]int, len(o.S))
			for i := range axes {
				axes[i] = i - len(o.S)
			}
		default:
			axes = make([]int, nd)
			for i := range axes {
				axes[i] = i
			}
		}
	}
	if r == oneD && len(axes) != 1 {
		return nil, false, fmt.Errorf("%w: a 1-D transform takes one axis, got %v", ErrArgs, axes)
	}
	if len(axes) == 0 {
		return nil, false, fmt.Errorf("%w: no axis to transform", ErrArgs)
	}
	seen := make([]bool, nd)
	out := make([]int, len(axes))
	for i, ax := range axes {
		n := ax
		if n < 0 {
			n += nd
		}
		if n < 0 || n >= nd {
			return nil, false, fmt.Errorf("%w: axis %d out of range for %d-dimensional array", ErrArgs, ax, nd)
		}
		dup = dup || seen[n]
		seen[n] = true
		out[i] = n
	}
	return out, dup, nil
}

// outLengths returns the length of each transformed axis of the result's
// real or complex signal (before r2c halves the last one).
func outLengths(a *ndarray.Array, o Options, k kind, axes []int) ([]int, error) {
	shape := a.Shape()
	if o.S != nil && len(o.S) != len(axes) {
		return nil, fmt.Errorf("%w: %d lengths for %d axes", ErrArgs, len(o.S), len(axes))
	}
	out := make([]int, len(axes))
	for i, ax := range axes {
		switch {
		case o.S != nil:
			out[i] = o.S[i]
		case (k == c2r || k == herm) && i == len(axes)-1:
			out[i] = 2 * (shape[ax] - 1)
		default:
			out[i] = shape[ax]
		}
		if out[i] < 1 {
			return nil, fmt.Errorf("%w: length %d along axis %d", ErrArgs, out[i], ax)
		}
	}
	return out, nil
}

// resize cuts or zero-pads a to length n along axis ax.
func resize(a *ndarray.Array, ax, n int) *ndarray.Array {
	shape := a.Shape()
	m := shape[ax]
	if n == m {
		return a
	}
	idx := make([]ndarray.Index, len(shape))
	for i := range idx {
		idx[i] = ndarray.All()
	}
	if n < m {
		idx[ax] = ndarray.To(n)
		v, _ := a.Slice(idx...) // a range within the axis: cannot fail
		return v
	}
	shape[ax] = n - m
	pad, _ := ndarray.ZerosOf(a.DType(), shape...) // a's shape with one axis grown: valid
	r, _ := ndarray.Concatenate([]*ndarray.Array{a, pad}, ax)
	return r
}

// complexTransform runs FFTN/IFFTN along axes.
func complexTransform(x *ndarray.Array, shape, axes []int, norm Norm, inverse, single bool) *ndarray.Array {
	o := gofft.Options{Axes: axes, Norm: norm}
	if single {
		d, _ := ndarray.Data[complex64](x.AsType(ndarray.Complex64))
		var r []complex64
		if inverse {
			r = gofft.IFFTN32With(d, shape, o)
		} else {
			r = gofft.FFTN32With(d, shape, o)
		}
		return wrap(r, shape)
	}
	d, _ := ndarray.Data[complex128](x.AsType(ndarray.Complex128))
	var r []complex128
	if inverse {
		r = gofft.IFFTNWith(d, shape, o)
	} else {
		r = gofft.FFTNWith(d, shape, o)
	}
	return wrap(r, shape)
}

// realForward runs RFFTN along axes; the last axis of the result holds
// n/2+1 bins.
func realForward(x *ndarray.Array, shape, axes []int, norm Norm, single bool) *ndarray.Array {
	o := gofft.Options{Axes: axes, Norm: norm}
	outShape := append([]int(nil), shape...)
	last := axes[len(axes)-1]
	outShape[last] = shape[last]/2 + 1
	if single {
		d, _ := ndarray.Data[float32](x.Real().AsType(ndarray.Float32))
		return wrap(gofft.RFFTN32With(d, shape, o), outShape)
	}
	d, _ := ndarray.Data[float64](x.Real().AsType(ndarray.Float64))
	return wrap(gofft.RFFTNWith(d, shape, o), outShape)
}

// realInverse runs IRFFTN along axes into a real array of the given shape.
func realInverse(x *ndarray.Array, shape, axes []int, norm Norm, single bool) *ndarray.Array {
	o := gofft.Options{Axes: axes, Norm: norm}
	if single {
		d, _ := ndarray.Data[complex64](x.AsType(ndarray.Complex64))
		return wrap(gofft.IRFFTN32With(d, shape, o), shape)
	}
	d, _ := ndarray.Data[complex128](x.AsType(ndarray.Complex128))
	return wrap(gofft.IRFFTNWith(d, shape, o), shape)
}

// wrap makes an array of a transform's output. The lengths always match.
func wrap[T float32 | float64 | complex64 | complex128](d []T, shape []int) *ndarray.Array {
	r, _ := ndarray.FromSlice(d, shape...)
	return r
}

// FFTFreq is numpy.fft.fftfreq: the sample frequencies of an n-point FFT with
// sample spacing d.
func FFTFreq(n int, d float64) (*ndarray.Array, error) {
	if n < 1 {
		return nil, fmt.Errorf("%w: fftfreq of %d points", ErrArgs, n)
	}
	return wrap(gofft.FFTFreq(n, d), []int{n}), nil
}

// RFFTFreq is numpy.fft.rfftfreq: the n/2+1 non-negative frequencies of an
// n-point RFFT.
func RFFTFreq(n int, d float64) (*ndarray.Array, error) {
	if n < 1 {
		return nil, fmt.Errorf("%w: rfftfreq of %d points", ErrArgs, n)
	}
	return wrap(gofft.RFFTFreq(n, d), []int{n/2 + 1}), nil
}

// FFTShift is numpy.fft.fftshift: it rotates each of the given axes (every
// axis when none is given) by half its length, moving the zero frequency to
// the centre.
func FFTShift(a *ndarray.Array, axes ...int) (*ndarray.Array, error) {
	return shift(a, axes, false)
}

// IFFTShift is numpy.fft.ifftshift, the inverse of FFTShift.
func IFFTShift(a *ndarray.Array, axes ...int) (*ndarray.Array, error) {
	return shift(a, axes, true)
}

// shift rolls each axis by n/2 (FFTShift) or -(n/2) (IFFTShift), for any
// dtype, by taking the two halves and joining them the other way round.
func shift(a *ndarray.Array, axes []int, inverse bool) (*ndarray.Array, error) {
	if axes == nil {
		axes = make([]int, a.Ndim())
		for i := range axes {
			axes[i] = i
		}
	}
	var norm []int
	if len(axes) > 0 {
		var err error
		if norm, _, err = resolveAxes(a, Options{Axes: axes}, allD); err != nil {
			return nil, err
		}
	}
	x := a
	shape := a.Shape()
	for _, ax := range norm {
		n := shape[ax]
		cut := n - n/2 // fftshift moves the last n/2 to the front
		if inverse {
			cut = n / 2
		}
		idx := make([]ndarray.Index, len(shape))
		for i := range idx {
			idx[i] = ndarray.All()
		}
		idx[ax] = ndarray.From(cut)
		tail, _ := x.Slice(idx...)
		idx[ax] = ndarray.To(cut)
		head, _ := x.Slice(idx...)
		x, _ = ndarray.Concatenate([]*ndarray.Array{tail, head}, ax)
	}
	if x == a {
		return a.Copy(), nil
	}
	return x, nil
}
