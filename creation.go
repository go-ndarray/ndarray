package ndarray

import "fmt"

// Linspace returns num evenly spaced samples over the closed interval
// [start, stop], matching numpy.linspace with the default endpoint=True. num
// must be non-negative; num == 0 yields an empty array and num == 1 yields
// [start]. The spacing is (stop-start)/(num-1) for num > 1.
func Linspace(start, stop float64, num int) (*Array, error) {
	if num < 0 {
		return nil, fmt.Errorf("%w: linspace num must be non-negative, got %d",
			ErrShapeMismatch, num)
	}
	data := make([]float64, num)
	switch {
	case num == 1:
		data[0] = start
	case num > 1:
		step := (stop - start) / float64(num-1)
		for i := 0; i < num; i++ {
			data[i] = start + float64(i)*step
		}
		// Pin the final sample exactly to stop, as numpy does, to avoid
		// floating-point drift at the endpoint.
		data[num-1] = stop
	}
	return FromData(data, num)
}

// Eye returns an array with ones on the k-th diagonal and zeros elsewhere,
// matching numpy.eye(n, m, k). A non-positive m defaults to n (a square
// matrix), the common case; pass m > 0 for a rectangular result. k > 0 selects
// a superdiagonal, k < 0 a subdiagonal. n must be non-negative.
func Eye(n, m, k int) (*Array, error) {
	if m <= 0 {
		m = n
	}
	a, err := New(n, m) // New validates n (and m) are non-negative
	if err != nil {
		return nil, err
	}
	for i := 0; i < n; i++ {
		j := i + k
		if j >= 0 && j < m {
			a.data[i*m+j] = 1
		}
	}
	return a, nil
}

// Identity returns the n-by-n identity matrix, matching numpy.identity(n).
func Identity(n int) (*Array, error) {
	return Eye(n, n, 0)
}
