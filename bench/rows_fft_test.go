package bench

import (
	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/fft"
)

// FFT rows. The reference is numpy.fft (pocketfft), which is what ndarray/fft
// documents itself against; go-fft/fft carries the scipy.fft comparison. The
// plan cost is inside both sides: ndarray/fft builds or fetches a plan per call,
// numpy.fft keeps its own plan cache warm after calibration. go-fft reads
// GOMAXPROCS once at init (go-fft/fft#107): run with GOMAXPROCS=1 in the
// environment.

const (
	f1 = 65_536 // 1-D length
	f2 = 256    // 2-D side
	f3 = 32     // 3-D side
)

var (
	fReal1  = dvec(-1, 1, f1, 0)
	fCplx1  = dcplx(f1)
	fCplxH1 = dcplx(f1/2 + 1) // the half spectrum IRFFT and HFFT take
	fReal2  = dmat(-1, 1, f2, f2, 0)
	fCplx2  = dcplx(f2, f2)
	fCplxH2 = dcplx(f2, f2/2+1)
	fReal3  = dreshape(dvec(-1, 1, f3*f3*f3, 0), f3, f3, f3)
	fCplx3  = dcplx(f3, f3, f3)
	fCplxH3 = dcplx(f3, f3, f3/2+1)
)

func fftRows() []row {
	var rs []row
	add := func(name, py string, in ds, g func(a *ndarray.Array) (*ndarray.Array, error), gw func(a *ndarray.Array, o fft.Options) (*ndarray.Array, error)) {
		rs = append(rs, row{name: "fft." + name, fn: "fft." + name, ref: "numpy np.fft." + py, args: []ds{in},
			call: "np.fft." + py,
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(g(a[0])) } }})
		rs = append(rs, row{name: "fft." + name + "With", fn: "fft." + name + "With", ref: "numpy np.fft." + py + "(norm='ortho')", args: []ds{in},
			call: "lambda x: np.fft." + py + "(x, norm='ortho')",
			body: func(a []*ndarray.Array) func() {
				return func() { sinkA = must(gw(a[0], fft.Options{Norm: fft.Ortho})) }
			}})
	}
	add("FFT", "fft", fCplx1, fft.FFT, fft.FFTWith)
	add("IFFT", "ifft", fCplx1, fft.IFFT, fft.IFFTWith)
	add("RFFT", "rfft", fReal1, fft.RFFT, fft.RFFTWith)
	add("IRFFT", "irfft", fCplxH1, fft.IRFFT, fft.IRFFTWith)
	add("HFFT", "hfft", fCplxH1, fft.HFFT, fft.HFFTWith)
	add("IHFFT", "ihfft", fReal1, fft.IHFFT, fft.IHFFTWith)
	add("FFT2", "fft2", fCplx2, fft.FFT2, fft.FFT2With)
	add("IFFT2", "ifft2", fCplx2, fft.IFFT2, fft.IFFT2With)
	add("RFFT2", "rfft2", fReal2, fft.RFFT2, fft.RFFT2With)
	add("IRFFT2", "irfft2", fCplxH2, fft.IRFFT2, fft.IRFFT2With)
	add("FFTN", "fftn", fCplx3, fft.FFTN, fft.FFTNWith)
	add("IFFTN", "ifftn", fCplx3, fft.IFFTN, fft.IFFTNWith)
	add("RFFTN", "rfftn", fReal3, fft.RFFTN, fft.RFFTNWith)
	add("IRFFTN", "irfftn", fCplxH3, fft.IRFFTN, fft.IRFFTNWith)
	rs = append(rs,
		row{name: "fft.FFTShift", fn: "fft.FFTShift", ref: "numpy np.fft.fftshift", args: []ds{fReal2}, call: "np.fft.fftshift",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(fft.FFTShift(a[0])) } }},
		row{name: "fft.IFFTShift", fn: "fft.IFFTShift", ref: "numpy np.fft.ifftshift", args: []ds{fReal2}, call: "np.fft.ifftshift",
			body: func(a []*ndarray.Array) func() { return func() { sinkA = must(fft.IFFTShift(a[0])) } }},
		row{name: "fft.FFTFreq", fn: "fft.FFTFreq", ref: "numpy np.fft.fftfreq", call: "lambda: np.fft.fftfreq(65536, 0.1)",
			body: func([]*ndarray.Array) func() { return func() { sinkA = must(fft.FFTFreq(f1, 0.1)) } }},
		row{name: "fft.RFFTFreq", fn: "fft.RFFTFreq", ref: "numpy np.fft.rfftfreq", call: "lambda: np.fft.rfftfreq(65536, 0.1)",
			body: func([]*ndarray.Array) func() { return func() { sinkA = must(fft.RFFTFreq(f1, 0.1)) } }},
	)
	return rs
}
