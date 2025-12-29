package parta

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

func extractSeedsFromString(line string) ([]uint64, bool) {

	numbers_string, number_string_found := strings.CutPrefix(line, "seeds: ")
	if !number_string_found {
		return nil, false
	}

	number_strings := strings.Split(numbers_string, " ")

	seeds := []uint64{}
	// for _, SeedMap := range sml.SeedMaps {

	for _, number_string := range number_strings {
		number, err := strconv.ParseUint(number_string, 10, 64)
		if err != nil {
			return nil, false
		}
		seeds = append(seeds, number)
	}

	return seeds, true
}

func PartA(input_file_path string) uint64 {

	// open file
	f, err := os.Open(input_file_path)
	if err != nil {
		log.Fatal(err)
	}
	// remember to close the file at the end of the program
	defer f.Close()

	// read the file line by line using scanner
	scanner := bufio.NewScanner(f)

	var seeds []uint64
	var seedMapLayers []seedmaplayer.SeedMapLayer
	var currSeedMapLayer *seedmaplayer.SeedMapLayer

	for scanner.Scan() {
		if strings.Compare(scanner.Text(), "") == 0 {
			continue
		}

		if strings.Contains(scanner.Text(), "seeds: ") {
			seeds, _ = extractSeedsFromString(scanner.Text())
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

	for _, seed := range seeds {

		if value := seedmaplayer.MapSeedInLayers(seedMapLayers, seed); value < min_value {
			min_value = value
		}

	}

	return min_value
}
