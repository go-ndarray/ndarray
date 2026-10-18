//go:build !amd64 && !arm64

package kernels

// addBlocks8F32 is the scalar loop on the architectures without a float32
// block kernel yet.
func addBlocks8F32(r *[8]float32, a []float32) { addBlocks8F32Go(r, a) }
