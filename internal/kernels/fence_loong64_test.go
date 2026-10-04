package kernels

// forEachFallback runs f again with the scalar kernel used without LASX.
func forEachFallback(f func()) {
	saved := useLASX
	useLASX = false
	defer func() { useLASX = saved }()
	f()
}
