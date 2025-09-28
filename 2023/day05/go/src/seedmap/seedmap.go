package seedmap

import (
	"strconv"
	"strings"
)

type SeedMap struct {
	SourceStart uint64
	SourceEnd   uint64
	TargetStart uint64
	TargetEnd   uint64
	Size        uint64
}

func SeedMapNew(source uint64, target uint64, size uint64) *SeedMap {
	sm := new(SeedMap)
	sm.SourceStart = source
	sm.SourceEnd = source + size
	sm.TargetStart = target
	sm.TargetEnd = target + size
	sm.Size = size

	return sm
}

func SeedMapNewFromString(seedMapString string) (*SeedMap, bool) {

	// number_string, number_string_found := strings.CutPrefix(seedMapString, "seeds: ")
	// if !number_string_found {
	// 	fmt.Println("blahs")
	// 	return nil, false
	// }

	number_strings := strings.Split(seedMapString, " ")

	if len(number_strings) != 3 {
		return nil, false
	}

	target, _ := strconv.ParseUint(number_strings[0], 10, 64)
	source, _ := strconv.ParseUint(number_strings[1], 10, 64)
	size, _ := strconv.ParseUint(number_strings[2], 10, 64)

	return SeedMapNew(source, target, size), true
}

func (sm *SeedMap) MapSeed(value uint64) (uint64, bool) {

	if (value >= sm.SourceStart) && (value < sm.SourceEnd) {
		return value - sm.SourceStart + sm.TargetStart, true
	}

	return value, false
}
