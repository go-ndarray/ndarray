package ndarray

import (
	"html"
	"math"
	"strconv"
	"strings"
)

// Printing, laid out the way NumPy prints: nested brackets, every element
// right-aligned to one width, lines broken at 75 columns, and, above 1000
// elements, only the first and last three items of each axis with "..."
// between them. Elements are written by the package's own formatting (the
// shortest form that reads back exactly), so the digits may differ from
// NumPy's, which pads every element to a common precision.

const (
	printThreshold = 1000 // above this many elements, summarise
	printEdge      = 3    // items kept at each end of a summarised axis
	printWidth     = 75   // line width
)

// Repr returns the array as NumPy's repr shows it: array([...]), with the
// dtype when it is not the default for its kind, and the shape when the
// elements are summarised.
func (a *Array) Repr() string {
	const open = "array("
	body := a.format(len(open))
	var b strings.Builder
	b.WriteString(open)
	b.WriteString(body)
	if a.Size() > printThreshold {
		b.WriteString(", shape=(")
		for i, d := range a.shape {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(strconv.Itoa(d))
		}
		if len(a.shape) == 1 {
			b.WriteByte(',')
		}
		b.WriteByte(')')
	}
	switch a.dtype {
	case Float64, Int64, Bool, Complex128:
	default:
		b.WriteString(", dtype=" + a.dtype.String())
	}
	b.WriteByte(')')
	return b.String()
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

// format lays out the elements, the first line starting at column indent.
func (a *Array) format(indent int) string {
	if len(a.shape) == 0 {
		return a.element(a.contiguousStore(), 0)
	}
	flat := a.contiguousStore()
	summarise := a.Size() > printThreshold
	width := 0
	a.visit(summarise, func(i int) {
		width = max(width, len(a.element(flat, i)))
	})
	var b strings.Builder
	a.formatAxis(&b, flat, 0, 0, indent, width, summarise)
	return b.String()
}

// visit calls f with the flat index of every element that is printed.
func (a *Array) visit(summarise bool, f func(int)) {
	var walk func(axis, base int)
	walk = func(axis, base int) {
		if axis == len(a.shape) {
			f(base)
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

// formatAxis writes the sub-array at flat offset base starting at axis.
func (a *Array) formatAxis(b *strings.Builder, flat any, axis, base, indent, width int, summarise bool) {
	b.WriteByte('[')
	stride := prod(a.shape[axis+1:])
	idx := shownIndices(a.shape[axis], summarise)
	last := axis == len(a.shape)-1
	col := indent + 1 // the column after the bracket
	for k, i := range idx {
		if last {
			s := "..."
			if i >= 0 {
				s = strings.Repeat(" ", width-len(a.element(flat, base+i))) + a.element(flat, base+i)
			}
			if k > 0 {
				b.WriteByte(',')
				col++
				if col+1+len(s)+1 > printWidth {
					b.WriteString("\n" + strings.Repeat(" ", indent+1))
					col = indent + 1
				} else {
					b.WriteByte(' ')
					col++
				}
			}
			b.WriteString(s)
			col += len(s)
			continue
		}
		if k > 0 {
			b.WriteString("," + strings.Repeat("\n", len(a.shape)-axis-1) + strings.Repeat(" ", indent+1))
		}
		if i < 0 {
			b.WriteString("...")
			continue
		}
		a.formatAxis(b, flat, axis+1, base+i*stride, indent+1, width, summarise)
	}
	b.WriteByte(']')
}

// element writes element i of a contiguous storage slice for printing:
// integral floats end in a point ("2."), as in NumPy.
func (a *Array) element(flat any, i int) string {
	s := elemString(flat, i)
	switch v := flat.(type) {
	case []float64:
		return pointed(s, v[i])
	case []float32:
		return pointed(s, float64(v[i]))
	case []bool:
		if v[i] {
			return "True"
		}
		return "False"
	case []complex64:
		return complexElement(complex128(v[i]), 32)
	case []complex128:
		return complexElement(v[i], 64)
	}
	return s
}

// complexElement writes a complex element as NumPy does inside an array:
// 1.+2.j, no parentheses.
func complexElement(z complex128, bits int) string {
	re, im := real(z), imag(z)
	sign := "+"
	if im < 0 || (im == 0 && math.Signbit(im)) {
		sign, im = "-", -im
	}
	return pointed(floatString(re, bits), re) + sign + pointed(floatString(im, bits), im) + "j"
}

// pointed adds NumPy's trailing point to an integral float written without
// an exponent.
func pointed(s string, v float64) string {
	if v == math.Trunc(v) && !math.IsInf(v, 0) && !strings.ContainsAny(s, "e.") {
		return s + "."
	}
	return s
}

// htmlTable writes a one- or two-dimensional array as a table, summarised
// like the text form.
func (a *Array) htmlTable() string {
	rows, cols := 1, a.shape[0]
	if len(a.shape) == 2 {
		rows, cols = a.shape[0], a.shape[1]
	}
	flat := a.contiguousStore()
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
			b.WriteString("<td>" + html.EscapeString(a.element(flat, i*cols+j)) + "</td>")
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
