package seedmaplayer

import (
	"example.com/day5/src/seedmap"
)

type SeedMapLayer struct {
	SeedMaps []seedmap.SeedMap
}

func (sml *SeedMapLayer) AddSeedMap(sm seedmap.SeedMap) {
	sml.SeedMaps = append(sml.SeedMaps, sm)
}

func (sml *SeedMapLayer) MapSeed(seedValue uint64) uint64 {
	for _, SeedMap := range sml.SeedMaps {
		var new_value, result = SeedMap.MapSeed(seedValue)
		if result {
			return new_value
		}
	}
	return seedValue
}

func MapSeedInLayers(smls []SeedMapLayer, seed_value uint64) uint64 {

	var value = seed_value
	for _, seedMapLayer := range smls {
		value = seedMapLayer.MapSeed(value)
	}

	return value

}
