package parta

import (
	"bufio"
	"log"
	"os"
)

func PartA(input_file_path string) uint64 {

	// open file
	f, err := os.Open(input_file_path)
	if err != nil {
		log.Fatal(err)
	}
	// remember to close the file at the end of the program
	defer f.Close()

	val := 5

	// read the file line by line using scanner
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {

		line_text := scanner.Text()
		// for index := 0; index < len(line_text); index++ {
		// 	println(line_text[index])
		// }
		for _, char := range line_text {
			switch char {
			case 'U':
				if 4 <= val {
					val -= 3
				}
				break
			case 'D':
				if 6 >= val {
					val += 3
				}
				break
			case 'L':
				if (val == 1) || (val == 4) || (val == 7) {
					break
				}
				val -= 1
				break
			case 'R':
				if (val == 3) || (val == 6) || (val == 9) {
					break
				}
				val += 1
				break
			}
		}

		print(val)
	}

	print("\n")
	return 1
}
