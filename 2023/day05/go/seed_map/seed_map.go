package seedmap

import (
	"errors"
)

type seedmap struct {
	source uint64
	target uint64
	size   uint64
}

func New(source uint64, target uint64, size uint64) seedmap {
	return seedmap{source: source, target: target, size: size}
}

// input string is typically "1 2 5\n"
func NewFromString(inputString string) (seedmap, error) {
	return seedmap{}, errors.New("invalid seed map string")
}

func (sm seedmap) MapSeed(seed_value *uint64) bool {
	if *seed_value < sm.source ||
		*seed_value >= (sm.source+sm.size) {
		return false
	}

	*seed_value = (*seed_value - sm.source) + sm.target
	return true
}
