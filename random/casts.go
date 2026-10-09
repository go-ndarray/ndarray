package random

import "unsafe"

// Signed integers are drawn as the unsigned integers of the same width, as
// numpy does (it writes through a pointer of the unsigned type); these views
// share the storage.

func asUint8(d []int8) []uint8 {
	return unsafe.Slice((*uint8)(unsafe.Pointer(unsafe.SliceData(d))), len(d))
}
func asUint16(d []int16) []uint16 {
	return unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(d))), len(d))
}
func asUint32(d []int32) []uint32 {
	return unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(d))), len(d))
}
func asUint64(d []int64) []uint64 {
	return unsafe.Slice((*uint64)(unsafe.Pointer(unsafe.SliceData(d))), len(d))
}
