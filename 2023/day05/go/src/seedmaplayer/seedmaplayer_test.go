package seedmaplayer

import (
	"testing"

	"example.com/day5/src/seedmap"
)

func TestMapSeed_in_layer_out_of_range_low(t *testing.T) {

	sml := new(SeedMapLayer)

	sml.AddSeedMap(*seedmap.SeedMapNew(2, 10, 5))
	sml.AddSeedMap(*seedmap.SeedMapNew(20, 50, 2))

	initial_value := uint64(1)
	new_value := sml.MapSeed(initial_value)

	if initial_value != new_value {
		t.Error("Error mapping seed in layer")
	}
}

func TestMapSeed_in_layer_out_of_range_high(t *testing.T) {

	sml := new(SeedMapLayer)

	sml.AddSeedMap(*seedmap.SeedMapNew(2, 10, 5))
	sml.AddSeedMap(*seedmap.SeedMapNew(20, 50, 2))

	initial_value := uint64(7)
	new_value := sml.MapSeed(initial_value)

	if initial_value != new_value {
		t.Error("Error mapping seed in layer")
	}
}

func TestMapSeed_in_range_1(t *testing.T) {

	sml := new(SeedMapLayer)

	sml.AddSeedMap(*seedmap.SeedMapNew(2, 10, 5))
	sml.AddSeedMap(*seedmap.SeedMapNew(20, 50, 2))

	initial_value := uint64(4)
	new_value := sml.MapSeed(initial_value)

	if new_value != 12 {
		t.Error("Error mapping seed in layer")
	}
}

func TestMapSeed_in_range_2(t *testing.T) {

	sml := new(SeedMapLayer)

	sml.AddSeedMap(*seedmap.SeedMapNew(2, 10, 5))
	sml.AddSeedMap(*seedmap.SeedMapNew(20, 50, 2))

	initial_value := uint64(20)
	new_value := sml.MapSeed(initial_value)

	if new_value != 50 {
		t.Error("Error mapping seed in layer")
	}
}
