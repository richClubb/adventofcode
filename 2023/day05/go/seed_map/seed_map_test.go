package seedmap

import (
	seedmap "day5/seed_map"
	"testing"
)

func TestMapSeed(t *testing.T) {
	test_seed_map := seedmap.New(1, 5, 10)

	seed_value := 1
	test_seed_map.MapSeed(&seed_value)

}
