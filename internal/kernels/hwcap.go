package kernels

import "encoding/binary"

// atHWCAP is the auxiliary-vector key of the hardware-capability bits the
// kernel exports (AT_HWCAP).
const atHWCAP = 16

// hwcapFrom returns AT_HWCAP from the contents of /proc/self/auxv: pairs of
// 64-bit words in the machine's byte order (key, value), ended by key 0 --
// big-endian on s390x, so the order must be native. It returns 0 if the
// key is absent or the data is short. A pure function, so the parsing is
// tested on every target even though only loong64, riscv64 and s390x use it.
func hwcapFrom(auxv []byte) uint64 {
	for len(auxv) >= 16 {
		k, v := binary.NativeEndian.Uint64(auxv), binary.NativeEndian.Uint64(auxv[8:])
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
