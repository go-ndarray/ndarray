package kernels

import "encoding/binary"

// atHWCAP is the auxiliary-vector key of the hardware-capability bits the
// kernel exports (AT_HWCAP).
const atHWCAP = 16

// hwcapFrom returns AT_HWCAP from the contents of /proc/self/auxv: pairs of
// little-endian 64-bit words (key, value), ended by key 0. It returns 0 if the
// key is absent or the data is short. A pure function, so the parsing is
// tested on every target even though only loong64 uses it today.
func hwcapFrom(auxv []byte) uint64 {
	for len(auxv) >= 16 {
		k, v := binary.LittleEndian.Uint64(auxv), binary.LittleEndian.Uint64(auxv[8:])
		if k == 0 {
			break
		}
		if k == atHWCAP {
			return v
		}
		auxv = auxv[16:]
	}
	return 0
}
