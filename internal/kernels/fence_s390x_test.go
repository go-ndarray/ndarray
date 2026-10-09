package kernels

// forEachFallback runs f again with the scalar kernels used without the vector facility.
func forEachFallback(f func()) {
	saved, savedE := useVX, useVXE
	useVX, useVXE = false, false
	defer func() { useVX, useVXE = saved, savedE }()
	f()
}
