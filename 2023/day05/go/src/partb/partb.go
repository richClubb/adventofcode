package partb

import (
	"bufio"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"example.com/day5/src/seedmap"
	"example.com/day5/src/seedmaplayer"
)

type SeedRange struct {
	start uint64
	size  uint64
}

func extractSeedRangesFromString(line string) ([]SeedRange, bool) {
	numbers_string, number_string_found := strings.CutPrefix(line, "seeds: ")
	if !number_string_found {
		return nil, false
	}

	number_strings := strings.Split(numbers_string, " ")

	seedRanges := []SeedRange{}

	for index := 0; index < len(number_strings); index += 2 {
		start, err := strconv.ParseUint(number_strings[index], 10, 64)
		if err != nil {
			return nil, false
		}
		size, err := strconv.ParseUint(number_strings[index+1], 10, 64)
		if err != nil {
			return nil, false
		}

		seedRanges = append(seedRanges, SeedRange{start: start, size: size})
	}

	return seedRanges, true
}

func findMinValueInSeedRange(seed_range SeedRange, seed_map_layers []seedmaplayer.SeedMapLayer) uint64 {

	var min_value uint64 = math.MaxUint64

	seedRangeStart := seed_range.start
	seedRangeEnd := seed_range.start + seed_range.size
	for seed := seedRangeStart; seed < seedRangeEnd; seed++ {
		value := seedmaplayer.MapSeedInLayers(seed_map_layers, seed)

		if value < min_value {
			min_value = value
		}
	}

	return min_value
}

func PartB(input_file_path string) uint64 {

	// open file
	f, err := os.Open(input_file_path)
	if err != nil {
		log.Fatal(err)
	}
	// remember to close the file at the end of the program
	defer f.Close()

	// read the file line by line using scanner
	scanner := bufio.NewScanner(f)

	var seedRanges []SeedRange
	var seedMapLayers []seedmaplayer.SeedMapLayer
	var currSeedMapLayer *seedmaplayer.SeedMapLayer

	for scanner.Scan() {
		if strings.Compare(scanner.Text(), "") == 0 {
			continue
		}

		if strings.Contains(scanner.Text(), "seeds: ") {
			seedRanges, _ = extractSeedRangesFromString(scanner.Text())
			continue
		}

		if strings.Contains(scanner.Text(), ":") {
			seedMapLayers = append(seedMapLayers, seedmaplayer.SeedMapLayer{})
			currSeedMapLayer = &seedMapLayers[len(seedMapLayers)-1]
			continue
		}

		// found seed map
		var currSeedMap, _ = seedmap.SeedMapNewFromString(scanner.Text())
		currSeedMapLayer.AddSeedMap(*currSeedMap)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	var min_value uint64 = math.MaxUint64

	for _, seedRange := range seedRanges {

		if range_min_value := findMinValueInSeedRange(seedRange, seedMapLayers); range_min_value < min_value {
			min_value = range_min_value
		}
	}

	return min_value
}
