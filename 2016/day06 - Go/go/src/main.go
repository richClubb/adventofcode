package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/richClubb/adventofcode/tree/main/2016/day06/go/src/parta"
	"github.com/richClubb/adventofcode/tree/main/2016/day06/go/src/partb"
)

func main() {

	// Define flags with name, default value, and usage description
	file_path := flag.String("i", "", "File path to run")
	run_type := flag.String("r", "part_a", "which run to do {part_a, part_b}")

	flag.Parse()

	fmt.Println("Advent of code 2016 - day 06")
	fmt.Println(*run_type)

	if strings.Compare(*run_type, "part_a") == 0 {

		part_a_result := parta.PartA(*file_path)
		fmt.Println("Part A result: {}", part_a_result)

	} else if strings.Compare(*run_type, "part_b") == 0 {

		part_b_result := partb.PartB(*file_path)
		fmt.Println("Part B result: {}", part_b_result)

	}
}
