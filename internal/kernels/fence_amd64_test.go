package kernels

// forEachFallback runs f again with the SSE2 kernels the FMA ones replace.
func forEachFallback(f func()) {
	saved := useFMA
	useFMA = false
	defer func() { useFMA = saved }()
	f()
}
