package kernels

// forEachFallback runs f again with the scalar kernels used without the vector facility.
func forEachFallback(f func()) {
	saved := useVX
	useVX = false
	defer func() { useVX = saved }()
	f()
}
