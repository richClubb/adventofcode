package seedmap

import "fmt"

type SeedMap struct {
	SourceStart uint64
	SourceEnd   uint64
	TargetStart uint64
	TargetEnd   uint64
	Size        uint64
}

func New(source uint64, target uint64, size uint64) *SeedMap {
	sm := new(SeedMap)
	sm.SourceStart = source
	sm.SourceEnd = source + size
	sm.TargetStart = target
	sm.TargetEnd = target + size
	sm.Size = size

	return sm
}

func NewFromString(seedMapString string) *SeedMap {
	sm := new(SeedMap)

	return sm
}

func (sm *SeedMap) MapSeed(value *uint64) bool {
	fmt.Println("Mapping seed")

	if (*value >= sm.SourceStart) && (*value < sm.SourceEnd) {
		*value = *value - sm.SourceStart + sm.TargetStart
		return true
	}

	return false
}
