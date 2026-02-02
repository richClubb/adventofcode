package partb

import (
	"bufio"
	"fmt"
	"log"
	"os"
)

func move_val(value uint8, instruction rune) uint8 {

	switch value {
	case 0x01:
		if 'D' == instruction {
			value = 3
		}
		break
	case 0x02:
		switch instruction {
		case 'L':
			value = 0x03
			break
		case 'D':
			value = 0x08
			break
		}
		break
	case 0x03:
		switch instruction {
		case 'U':
			value = 0x1
			break
		case 'D':
			value = 0x7
			break
		case 'L':
			value = 0x02
			break
		case 'R':
			value = 0x04
			break
		}
		break
	case 0x04:
		switch instruction {
		case 'R':
			value = 0x03
			break
		case 'D':
			value = 0x06
			break
		}
		break
	case 0x05:
		if 'R' == instruction {
			value = 6
		}
		break
	case 0x06:
		switch instruction {
		case 'U':
			value = 0x2
			break
		case 'D':
			value = 0xA
			break
		case 'L':
			value = 0x05
			break
		case 'R':
			value = 0x07
			break
		}
		break
	case 0x07:
		switch instruction {
		case 'U':
			value = 0x3
			break
		case 'D':
			value = 0xB
			break
		case 'L':
			value = 0x06
			break
		case 'R':
			value = 0x08
			break
		}
		break
	case 0x08:
		switch instruction {
		case 'U':
			value = 0x4
			break
		case 'D':
			value = 0xC
			break
		case 'L':
			value = 0x07
			break
		case 'R':
			value = 0x09
			break
		}
		break
	case 0x09:
		if 'L' == instruction {
			value = 6
		}
		break
	case 0x0A:
		switch instruction {
		case 'R':
			value = 0x06
			break
		case 'U':
			value = 0x0B
			break
		}
		break
	case 0x0B:
		switch instruction {
		case 'U':
			value = 0x7
			break
		case 'D':
			value = 0xD
			break
		case 'L':
			value = 0x0A
			break
		case 'R':
			value = 0x0C
			break
		}
		break
	case 0x0C:
		switch instruction {
		case 'L':
			value = 0x0B
			break
		case 'U':
			value = 0x08
			break
		}
		break
	case 0x0D:
		if 'U' == instruction {
			value = 0x0B
		}
		break
	}
	return value
}

func PartB(input_file_path string) uint64 {

	// open file
	f, err := os.Open(input_file_path)
	if err != nil {
		log.Fatal(err)
	}
	// remember to close the file at the end of the program
	defer f.Close()

	var val uint8 = 5

	// read the file line by line using scanner
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {

		line_text := scanner.Text()
		// for index := 0; index < len(line_text); index++ {
		// 	println(line_text[index])
		// }
		for _, char := range line_text {
			val = move_val(val, char)
		}

		fmt.Printf("%X", val)
	}

	print("\n")
	return 1
}
