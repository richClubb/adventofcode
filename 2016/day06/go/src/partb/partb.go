package partb

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

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
	var data []string

	for scanner.Scan() {
		line_text := scanner.Text()
		data = append(data, line_text)
	}

	for index, _ := range data[0] {
		var char_map map[byte]int
		char_map = make(map[byte]int)

		for _, entry := range data {
			character := entry[index]

			val, ok := char_map[character]
			if ok {
				char_map[character] = val + 1
			} else {
				char_map[character] = 1
			}
		}

		var most_freq_char byte = 0
		var most_freq_char_count = 255
		for key, data := range char_map {
			if data < most_freq_char_count {
				most_freq_char_count = data
				most_freq_char = key
			}
		}

		fmt.Printf("%c", most_freq_char)
	}

	fmt.Printf("\n")

	return 1
}
