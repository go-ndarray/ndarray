package kernels

// forEachFallback runs f again with the scalar kernels used without V.
func forEachFallback(f func()) {
	saved := useRVV
	useRVV = false
	defer func() { useRVV = saved }()
	f()
}
