package ndarray

import (
	"html"
	"math"
	"strconv"
	"strings"
)

// Printing, as NumPy 2.5.3 prints with its default options: a port of
// numpy/_core/arrayprint.py (_array_repr_implementation, array2string,
// _formatArray, _extendLine, FloatingFormat, ComplexFloatingFormat,
// IntegerFormat, BoolFormat) and of the parts of dragon4.c those use
// (positional and scientific output, unique digits, a precision cut-off
// rounded half to even on the exact binary value, as strconv rounds).
//
// The defaults are NumPy's: precision 8 in "maxprec" mode, 75 columns,
// threshold 1000 elements, 3 edge items, sign "-", no small-number
// suppression. The format of each element is decided from the elements that
// are printed, as NumPy decides it from _leading_trailing(data).

const (
	printThreshold = 1000 // above this many elements, summarise
	printEdge      = 3    // items kept at each end of a summarised axis
	printWidth     = 75   // line width
	printPrecision = 8    // fraction digits at most, in "maxprec" mode
)

// Repr returns the array as NumPy's repr shows it: array([...]), followed by
// the shape when the elements are summarised or the array is empty, and the
// dtype when the values do not imply it (float64, int64, bool and complex128
// do) or the array is empty.
func (a *Array) Repr() string {
	const prefix = "array("
	var lst string
	if a.Size() == 0 {
		lst = "[]"
	} else {
		lst = a.formatArray(a.elementFormatter(), printWidth-len(")"), strings.Repeat(" ", len(prefix)+1))
	}
	var extras []string
	if (a.Size() == 0 && !(len(a.shape) == 1 && a.shape[0] == 0)) || a.Size() > printThreshold {
		extras = append(extras, "shape="+shapeString(a.shape))
	}
	if !dtypeImplied(a.dtype) || a.Size() == 0 {
		extras = append(extras, "dtype="+a.dtype.String())
	}
	if len(extras) == 0 {
		return prefix + lst + ")"
	}
	arr := prefix + lst + ","
	extra := strings.Join(extras, ", ") + ")"
	spacer := " "
	if lastLine := len(arr) - (strings.LastIndexByte(arr, '\n') + 1); lastLine+len(extra)+1 > printWidth {
		spacer = "\n" + strings.Repeat(" ", len(prefix))
	}
	return arr + spacer + extra
}

// dtypeImplied reports whether NumPy's repr leaves the dtype out because the
// printed values imply it.
func dtypeImplied(d DType) bool {
	switch d {
	case Float64, Int64, Bool, Complex128:
		return true
	}
	return false
}

// Display shows the array in a notebook: its Repr as text, and as an HTML
// table for one and two dimensions.
func (a *Array) Display() map[string][]byte {
	out := map[string][]byte{"text/plain": []byte(a.Repr())}
	if len(a.shape) == 1 || len(a.shape) == 2 {
		out["text/html"] = []byte(a.htmlTable())
	}
	return out
}

// shownIndices lists the indices of an axis of length n that are printed,
// with -1 where the "..." goes.
func shownIndices(n int, summarise bool) []int {
	if !summarise || n <= 2*printEdge {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	out := []int{0, 1, 2, -1}
	for i := n - printEdge; i < n; i++ {
		out = append(out, i)
	}
	return out
}

// shownFlat lists the flat (row-major) indices of the elements that are
// printed: NumPy's _leading_trailing.
func (a *Array) shownFlat(summarise bool) []int {
	var out []int
	var walk func(axis, base int)
	walk = func(axis, base int) {
		if axis == len(a.shape) {
			out = append(out, base)
			return
		}
		stride := prod(a.shape[axis+1:])
		for _, i := range shownIndices(a.shape[axis], summarise) {
			if i >= 0 {
				walk(axis+1, base+i*stride)
			}
		}
	}
	walk(0, 0)
	return out
}

// elementFormatter returns the function that writes element i of the
// contiguous storage, padded to the common width, built from the elements
// that are printed (NumPy's _get_format_function).
func (a *Array) elementFormatter() func(flat any, i int) string {
	flat := a.contiguousStore()
	shown := a.shownFlat(a.Size() > printThreshold)
	switch v := flat.(type) {
	case []float64:
		f := newFloatFormat(pick(v, shown, func(x float64) float64 { return x }), 64, '-')
		return func(s any, i int) string { return f.format(s.([]float64)[i]) }
	case []float32:
		f := newFloatFormat(pick(v, shown, func(x float32) float64 { return float64(x) }), 32, '-')
		return func(s any, i int) string { return f.format(float64(s.([]float32)[i])) }
	case []complex128:
		c := newComplexFormat(pick(v, shown, func(z complex128) complex128 { return z }), 64)
		return func(s any, i int) string { return c.format(s.([]complex128)[i]) }
	case []complex64:
		c := newComplexFormat(pick(v, shown, func(z complex64) complex128 { return complex128(z) }), 32)
		return func(s any, i int) string { return c.format(complex128(s.([]complex64)[i])) }
	case []bool:
		// BoolFormat: " True" so that it lines up with "False", except in a
		// 0-d array.
		t := " True"
		if len(a.shape) == 0 {
			t = "True"
		}
		return func(s any, i int) string {
			if s.([]bool)[i] {
				return t
			}
			return "False"
		}
	}
	// IntegerFormat: right-aligned to the longer of the largest and the
	// smallest printed value.
	width := 0
	for _, i := range shown {
		width = max(width, len(elemString(flat, i)))
	}
	return func(s any, i int) string {
		e := elemString(s, i)
		return strings.Repeat(" ", width-len(e)) + e
	}
}

// pick converts the elements of v at the indices idx.
func pick[T, U any](v []T, idx []int, conv func(T) U) []U {
	out := make([]U, len(idx))
	for k, i := range idx {
		out[k] = conv(v[i])
	}
	return out
}

// formatArray lays the array out as NumPy's _formatArray does, with ", " as
// the separator; lineWidth already excludes the closing ")".
func (a *Array) formatArray(format func(any, int) string, lineWidth int, nextLinePrefix string) string {
	flat := a.contiguousStore()
	summarise := a.Size() > printThreshold
	var rec func(axis, base int, hanging string, width int) string
	rec = func(axis, base int, hanging string, width int) string {
		if axis == len(a.shape) {
			return format(flat, base)
		}
		nextHanging := hanging + " "
		nextWidth := width - len("]")
		stride := prod(a.shape[axis+1:])
		idx := shownIndices(a.shape[axis], summarise)
		var s strings.Builder
		if axis == len(a.shape)-1 {
			elemWidth := width - max(len(","), len("]"))
			line := hanging
			for k, i := range idx {
				if k > 0 {
					line += ", "
				}
				word := "..."
				if i >= 0 {
					word = rec(axis+1, base+i*stride, nextHanging, nextWidth)
				}
				line = extendLine(&s, line, word, elemWidth, hanging)
			}
			s.WriteString(line)
		} else {
			sep := "," + strings.Repeat("\n", len(a.shape)-axis-1)
			for k, i := range idx {
				if k > 0 {
					s.WriteString(sep)
				}
				s.WriteString(hanging)
				if i < 0 {
					s.WriteString("...")
					continue
				}
				s.WriteString(rec(axis+1, base+i*stride, nextHanging, nextWidth))
			}
		}
		return "[" + s.String()[len(hanging):] + "]"
	}
	return rec(0, 0, nextLinePrefix, lineWidth)
}

// extendLine is NumPy's _extendLine: start a new line when the word would
// pass the width, unless the line holds nothing but its prefix (a wrap that
// would not help).
func extendLine(s *strings.Builder, line, word string, width int, prefix string) string {
	if len(line)+len(word) > width && len(line) > len(prefix) {
		s.WriteString(strings.TrimRight(line, " ") + "\n")
		line = prefix
	}
	return line + word
}

// floatFormat is NumPy's FloatingFormat in "maxprec" mode with precision 8.
type floatFormat struct {
	bits              int  // 64 or 32: the shortest digits are those of this type
	sign              byte // '-' (only negatives) or '+' (always)
	exp               bool // scientific notation
	precision         int  // fraction digits: the cut-off, or the common width in exp mode
	minDigits         int  // fraction digits padded with zeros up to this
	padLeft, padRight int
	expDigits         int
	nonFinites        bool
}

func newFloatFormat(vals []float64, bits int, sign byte) *floatFormat {
	f := &floatFormat{bits: bits, sign: sign, precision: printPrecision}
	var finite []float64
	maxAbs, minAbs := 0.0, math.Inf(1)
	for _, x := range vals {
		if math.IsInf(x, 0) || math.IsNaN(x) {
			f.nonFinites = true
			continue
		}
		finite = append(finite, x)
		if x != 0 {
			maxAbs, minAbs = max(maxAbs, math.Abs(x)), min(minAbs, math.Abs(x))
		}
	}
	if maxAbs > 0 {
		// The cut-off is 10**min(8, finfo(dtype).precision): 1e8 for
		// float64, 1e6 for float32 (whose decimal precision is 6).
		cutoff := 1e8
		if bits == 32 {
			cutoff = 1e6
		}
		f.exp = maxAbs >= cutoff || minAbs < 1e-4 || maxAbs/minAbs > 1000
	}
	f.expDigits = -1
	switch {
	case len(finite) == 0:
	case f.exp:
		// First pass: unique digits, trimmed, to learn the widths.
		for _, x := range finite {
			intPart, frac, e := f.scientific(x, printPrecision, 0, false)
			f.padLeft = max(f.padLeft, len(intPart))
			f.minDigits = max(f.minDigits, len(frac))
			f.expDigits = max(f.expDigits, len(e)-1)
		}
		f.precision = f.minDigits
		f.padRight = f.expDigits + 2 + f.precision
	default:
		for _, x := range finite {
			intPart, frac := f.positional(x)
			f.padLeft = max(f.padLeft, len(intPart))
			f.padRight = max(f.padRight, len(frac))
		}
	}
	if f.nonFinites {
		neginf := 0
		for _, x := range vals {
			if sign != '-' || math.IsInf(x, -1) {
				neginf = 1
			}
		}
		offset := f.padRight + 1
		f.padLeft = max(f.padLeft, len("nan")-offset, len("inf")+neginf-offset)
	}
	return f
}

// signed writes the sign NumPy's dragon4 writes: "-" for a negative value
// (negative zero included), "+" for any other when the mode is '+'.
func (f *floatFormat) signed(x float64) (string, float64) {
	if math.Signbit(x) {
		return "-", -x
	}
	if f.sign == '+' {
		return "+", x
	}
	return "", x
}

// positional returns the integer part (with its sign) and the fraction
// digits of x: the shortest digits that read back as x, cut to the precision
// (rounded half to even on the exact value), trailing zeros trimmed.
func (f *floatFormat) positional(x float64) (string, string) {
	sign, x := f.signed(x)
	s := strconv.FormatFloat(x, 'f', -1, f.bits)
	if dot := strings.IndexByte(s, '.'); dot >= 0 && len(s)-dot-1 > f.precision {
		s = strings.TrimRight(strconv.FormatFloat(x, 'f', f.precision, f.bits), "0")
	}
	intPart, frac, _ := strings.Cut(s, ".")
	return sign + intPart, frac
}

// scientific returns the mantissa's integer digit (with its sign), its
// fraction digits and the exponent ("+05"): unique digits cut to precision,
// trimmed, then padded with zeros to minDigits; the exponent has at least
// two digits, or expDigits when pad is set.
func (f *floatFormat) scientific(x float64, precision, minDigits int, pad bool) (string, string, string) {
	sign, x := f.signed(x)
	s := strconv.FormatFloat(x, 'e', -1, f.bits)
	mant, e, _ := strings.Cut(s, "e")
	if _, frac, _ := strings.Cut(mant, "."); len(frac) > precision {
		mant, e, _ = strings.Cut(strconv.FormatFloat(x, 'e', precision, f.bits), "e")
		mant = strings.TrimRight(mant, "0")
	}
	intPart, frac, _ := strings.Cut(mant, ".")
	frac += strings.Repeat("0", max(0, minDigits-len(frac)))
	digits := strings.TrimLeft(e[1:], "0")
	width := 2
	if pad {
		width = max(width, min(f.expDigits, 5))
	}
	digits = strings.Repeat("0", max(0, width-len(digits))) + digits
	return sign + intPart, frac, e[:1] + digits
}

// format writes x padded to the format's width.
func (f *floatFormat) format(x float64) string {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		var r string
		switch {
		case math.IsNaN(x):
			r = "nan"
			if f.sign == '+' {
				r = "+nan"
			}
		case x < 0:
			r = "-inf"
		case f.sign == '+':
			r = "+inf"
		default:
			r = "inf"
		}
		return strings.Repeat(" ", max(0, f.padLeft+f.padRight+1-len(r))) + r
	}
	if f.exp {
		intPart, frac, e := f.scientific(x, f.precision, f.minDigits, true)
		return strings.Repeat(" ", max(0, f.padLeft-len(intPart))) + intPart + "." + frac + "e" + e
	}
	intPart, frac := f.positional(x)
	return strings.Repeat(" ", max(0, f.padLeft-len(intPart))) + intPart + "." + frac + strings.Repeat(" ", max(0, f.padRight-len(frac)))
}

// complexFormat is NumPy's ComplexFloatingFormat: the real parts and the
// imaginary parts each get their own FloatingFormat, the imaginary one
// always signed, and the "j" goes before the imaginary part's padding.
type complexFormat struct{ re, im *floatFormat }

func newComplexFormat(vals []complex128, bits int) *complexFormat {
	re := make([]float64, len(vals))
	im := make([]float64, len(vals))
	for i, z := range vals {
		re[i], im[i] = real(z), imag(z)
	}
	return &complexFormat{newFloatFormat(re, bits, '-'), newFloatFormat(im, bits, '+')}
}

func (c *complexFormat) format(z complex128) string {
	i := c.im.format(imag(z))
	sp := len(strings.TrimRight(i, " "))
	return c.re.format(real(z)) + i[:sp] + "j" + i[sp:]
}

// htmlTable writes a one- or two-dimensional array as a table, summarised
// like the text form, each cell the element as Repr writes it, unpadded.
func (a *Array) htmlTable() string {
	rows, cols := 1, a.shape[0]
	if len(a.shape) == 2 {
		rows, cols = a.shape[0], a.shape[1]
	}
	flat := a.contiguousStore()
	format := a.elementFormatter()
	summarise := a.Size() > printThreshold
	var b strings.Builder
	b.WriteString(`<table class="ndarray"><caption>`)
	b.WriteString(html.EscapeString(shapeString(a.shape) + " " + a.dtype.String()))
	b.WriteString("</caption>")
	ci := shownIndices(cols, summarise)
	if len(a.shape) == 2 {
		b.WriteString("<thead><tr><th></th>")
		for _, j := range ci {
			b.WriteString("<th>" + index(j) + "</th>")
		}
		b.WriteString("</tr></thead>")
	}
	b.WriteString("<tbody>")
	for _, i := range shownIndices(rows, summarise && len(a.shape) == 2) {
		b.WriteString("<tr>")
		if len(a.shape) == 2 {
			b.WriteString("<th>" + index(i) + "</th>")
		}
		for _, j := range ci {
			if i < 0 || j < 0 {
				b.WriteString("<td>⋯</td>")
				continue
			}
			b.WriteString("<td>" + html.EscapeString(strings.TrimSpace(format(flat, i*cols+j))) + "</td>")
		}
		b.WriteString("</tr>")
	}
	b.WriteString("</tbody></table>")
	return b.String()
}

// index writes a row or column header, "⋯" for the summarised gap.
func index(i int) string {
	if i < 0 {
		return "⋯"
	}
	return strconv.Itoa(i)
}

// shapeString writes a shape as NumPy does: (3, 4), (5,), ().
func shapeString(shape []int) string {
	var b strings.Builder
	b.WriteByte('(')
	for i, d := range shape {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(d))
	}
	if len(shape) == 1 {
		b.WriteByte(',')
	}
	b.WriteByte(')')
	return b.String()
}
