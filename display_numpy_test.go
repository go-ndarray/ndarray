package ndarray

import (
	"math"
	"strconv"
	"testing"
)

// TestReprMatchesNumPysOwnTests checks Repr against the expected strings of
// NumPy's own test suite, numpy/_core/tests/test_arrayprint.py at tag v2.5.3,
// for every case that uses the default print options and a dtype this package
// has (no float16, no structured or string arrays). The strings are copied
// verbatim; the test each comes from is named beside it.
func TestReprMatchesNumPysOwnTests(t *testing.T) {
	f64 := func(v ...float64) *Array { return must(FromData(v, len(v))) }
	scalar := func(v float64) *Array { return must(FromData([]float64{v})) }
	nan, inf := math.NaN(), math.Inf(1)
	// test_floatmode: y, the float64 case, and w, built from the strings
	// "1e0" ... "1e24" (1e23 lies between two float64 values).
	y := f64(0.2918820979355541, 0.5064172631089138, 0.2848750619642916, 0.4342965294660567,
		0.7326538397312751, 0.3459503329096204, 0.0862072768214508, 0.39112753029631175)
	var w []float64
	for i := range 25 {
		v, err := strconv.ParseFloat("1e"+strconv.Itoa(i), 64)
		if err != nil {
			t.Fatal(err)
		}
		w = append(w, v)
	}
	c := must(FromSlice([]complex128{1 + 1i, 1.123456789 + 1.123456789i}, 2))
	cases := []struct {
		test string
		a    *Array
		want string
	}{
		{"test_nan_inf", f64(nan, inf), "array([nan, inf])"},
		{"test_float_spacing", f64(1, 2, 3), "array([1., 2., 3.])"},
		{"test_float_spacing", f64(1, 2, -10), "array([  1.,   2., -10.])"},
		{"test_float_spacing", scalar(1), "array(1.)"},
		{"test_float_spacing", scalar(-10), "array(-10.)"},
		{"test_float_spacing", f64(100, 2, -1), "array([100.,   2.,  -1.])"},
		{"test_float_spacing", f64(-100, 2, 1), "array([-100.,    2.,    1.])"},
		{"test_float_spacing", f64(nan, -inf), "array([ nan, -inf])"},
		{"test_bool_spacing", must(FromSlice([]bool{true, true}, 2)), "array([ True,  True])"},
		{"test_bool_spacing", must(FromSlice([]bool{true, false}, 2)), "array([ True, False])"},
		{"test_bool_spacing", must(FromSlice([]bool{true}, 1)), "array([ True])"},
		{"test_bool_spacing", must(FromSlice([]bool{true})), "array(True)"},
		{"test_bool_spacing", must(FromSlice([]bool{false})), "array(False)"},
		{"test_sign_spacing", must(Arange(0, 4, 1)), "array([0., 1., 2., 3.])"},
		{"test_sign_spacing", f64(1.234e9), "array([1.234e+09])"},
		{"test_sign_spacing", f64(0), "array([0.])"},
		{"test_sign_spacing", c, "array([1.        +1.j        , 1.12345679+1.12345679j])"},
		{"test_sign_spacing", f64(0, math.Copysign(0, -1)), "array([ 0., -0.])"},
		{"test_floatmode (maxprec)", y, "array([0.2918821 , 0.50641726, 0.28487506, 0.43429653, 0.73265384,\n" +
			"       0.34595033, 0.08620728, 0.39112753])"},
		{"test_floatmode (maxprec)", f64(w[0], w[5], w[10], w[15], w[20]), "array([1.e+00, 1.e+05, 1.e+10, 1.e+15, 1.e+20])"},
		{"test_floatmode (maxprec)", f64(1.234e1, 1e2, 1e123), "array([1.234e+001, 1.000e+002, 1.000e+123])"},
		// The unique-mode string for w: maxprec gives the same digits for
		// powers of ten, so this checks the wrapping at 75 columns.
		{"test_floatmode (w)", f64(w...), "array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,\n" +
			"       1.e+08, 1.e+09, 1.e+10, 1.e+11, 1.e+12, 1.e+13, 1.e+14, 1.e+15,\n" +
			"       1.e+16, 1.e+17, 1.e+18, 1.e+19, 1.e+20, 1.e+21, 1.e+22, 1.e+23,\n" +
			"       1.e+24])"},
		{"test_summarize_1d", must(Arange(0, 1001, 1)).AsType(Int64), "array([   0,    1,    2, ...,  998,  999, 1000], shape=(1001,))"},
		{"test_summarize_2d", must(must(Arange(0, 1002, 1)).AsType(Int64).Reshape(2, 501)),
			"array([[   0,    1,    2, ...,  498,  499,  500],\n" +
				"       [ 501,  502,  503, ...,  999, 1000, 1001]], shape=(2, 501))"},
		{"test_summarize_2d_dtype", must(must(Arange(0, 1002, 1)).AsType(Int16).Reshape(2, 501)),
			"array([[   0,    1,    2, ...,  498,  499,  500],\n" +
				"       [ 501,  502,  503, ...,  999, 1000, 1001]],\n" +
				"      shape=(2, 501), dtype=int16)"},
	}
	for _, tc := range cases {
		if got := tc.a.Repr(); got != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.test, got, tc.want)
		}
	}
}

// TestComplexSpecialValuesMatchNumPy is TestComplexArray.test_str (NumPy
// v2.5.3): every pair of 0, 1, -1, inf, -inf, nan as real and imaginary part,
// in a one-element array, for complex64 and complex128. NumPy's expected
// strings are str() forms, "[...]"; for one element repr shows the same
// between "array(" and ")".
func TestComplexSpecialValuesMatchNumPy(t *testing.T) {
	r := []float64{0, 1, -1, math.Inf(1), math.Inf(-1), math.NaN()}
	wanted := []string{
		"[0.+0.j]", "[0.+1.j]", "[0.-1.j]", "[0.+infj]", "[0.-infj]", "[0.+nanj]",
		"[1.+0.j]", "[1.+1.j]", "[1.-1.j]", "[1.+infj]", "[1.-infj]", "[1.+nanj]",
		"[-1.+0.j]", "[-1.+1.j]", "[-1.-1.j]", "[-1.+infj]", "[-1.-infj]", "[-1.+nanj]",
		"[inf+0.j]", "[inf+1.j]", "[inf-1.j]", "[inf+infj]", "[inf-infj]", "[inf+nanj]",
		"[-inf+0.j]", "[-inf+1.j]", "[-inf-1.j]", "[-inf+infj]", "[-inf-infj]", "[-inf+nanj]",
		"[nan+0.j]", "[nan+1.j]", "[nan-1.j]", "[nan+infj]", "[nan-infj]", "[nan+nanj]",
	}
	k := 0
	for _, re := range r {
		for _, im := range r {
			want := "array(" + wanted[k] + ")"
			if got := must(FromSlice([]complex128{complex(re, im)}, 1)).Repr(); got != want {
				t.Errorf("complex128(%v, %v): got %q, want %q", re, im, got, want)
			}
			want = "array(" + wanted[k] + ", dtype=complex64)"
			if got := must(FromSlice([]complex64{complex64(complex(re, im))}, 1)).Repr(); got != want {
				t.Errorf("complex64(%v, %v): got %q, want %q", re, im, got, want)
			}
			k++
		}
	}
}

// TestReprRulesWithoutANumPyCase covers the paths NumPy's suite has no
// default-options case for, with the expectation worked out from the rules of
// arrayprint.py v2.5.3 (each stated beside it).
func TestReprRulesWithoutANumPyCase(t *testing.T) {
	for _, tc := range []struct {
		a    *Array
		want string
		rule string
	}{
		{must(Zeros(0)), "array([], dtype=float64)",
			"an empty array prints [] and always its dtype; shape (0,) is implied"},
		{must(Zeros(0, 3)), "array([], shape=(0, 3), dtype=float64)",
			"an empty array of any other shape also prints its shape"},
		{must(FromData([]float64{1.123456789e9}, 1)), "array([1.12345679e+09])",
			"scientific mantissas are cut to 8 fraction digits, rounded"},
		{must(FromData([]float64{-1e-5, 2}, 2)), "array([-1.e-05,  2.e+00])",
			"min |x| < 1e-4 switches to scientific; the sign widens the integer part"},
	} {
		if got := tc.a.Repr(); got != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.rule, got, tc.want)
		}
	}
}

// TestElementsMatchNumPyGh28679 checks the elements of test_gh_28679 (NumPy
// v2.5.3), whose expected strings are str() forms: the elements are the same
// as repr's, only the separator differs, so each element is compared alone.
// They pin the exponent cut-off that depends on the dtype: 1e6 for float32.
func TestElementsMatchNumPyGh28679(t *testing.T) {
	elements := func(a *Array) []string {
		f := a.elementFormatter()
		flat := a.contiguousStore()
		out := make([]string, a.Size())
		for i := range out {
			out[i] = f(flat, i)
		}
		return out
	}
	for _, tc := range []struct {
		a    *Array
		want []string
	}{
		{must(FromSlice([]float32{999999, 999999}, 2)), []string{"999999.", "999999."}},
		{must(FromSlice([]float32{999999, -1000000}, 2)), []string{" 9.99999e+05", "-1.00000e+06"}},
		{must(FromSlice([]complex64{999999 + 999999i, 999999 + 999999i}, 2)), []string{"999999.+999999.j", "999999.+999999.j"}},
		{must(FromSlice([]complex64{999999 + 999999i, complex(999999, -1000000)}, 2)), []string{"999999.+9.99999e+05j", "999999.-1.00000e+06j"}},
	} {
		got := elements(tc.a)
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%v: element %d = %q, want %q", tc.a.dtype, i, got[i], tc.want[i])
			}
		}
	}
}
