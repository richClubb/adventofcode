package main

import (
	"flag"
	"fmt"
	"strings"

	"example.com/day5/src/parta"
	"example.com/day5/src/partb"
)

func main() {

	// Define flags with name, default value, and usage description
	file_path := flag.String("i", "", "File path to run")
	run_type := flag.String("r", "part_a", "which run to do {part_a, part_b}")

	flag.Parse()

	fmt.Println("Advent of code 2023 - day 05")

	if strings.Compare(*run_type, "part_a") == 0 {

		part_a_result := parta.PartA(*file_path)
		fmt.Println("Part A result: {}", part_a_result)

	} else if strings.Compare(*run_type, "part_b") == 0 {

		part_b_result := partb.PartB(*file_path)
		fmt.Println("Part B result: {}", part_b_result)

	}
}
