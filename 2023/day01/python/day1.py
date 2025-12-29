#!/bin/env python3

import argparse
import os
from functools import reduce

number_words = {
    "one": "1ne",
    "two": "2wo",
    "three": "3hree",
    "four": "4our",
    "five": "5ive",
    "six": "6ix",
    "seven": "7even",
    "eight": "8ight",
    "nine": "9ine",
}


def is_number(char):
    if (ord(char) >= 48) and (ord(char) <= 57):
        return True
    return False


def first_last_num(input_list):
    if len(input_list) == 0:
        return int(f"{input_list[0]}{input_list[0]}")
    return int(f"{input_list[0]}{input_list[-1]}")


def convert_words_to_numbers(input_string):
    curr_string = ""

    for pos, char in enumerate(input_string):
        curr_string += char

        for number_word, number_str in number_words.items():
            if number_word in curr_string:
                curr_string = curr_string.replace(number_word, number_str)

    return curr_string


def process_file_a(path):
    with open(path) as file:
        total = sum(
            map(
                lambda x: first_last_num(list(filter(is_number, x))),
                file.readlines(),
            )
        )

        return total


def process_file_b(path):
    with open(path) as file:
        total = sum(
            map(
                lambda x: first_last_num(
                    list(filter(is_number, convert_words_to_numbers(x)))
                ),
                file.readlines(),
            )
        )
        return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    print(process_file_a(args.input_file_path))
    print(process_file_b(args.input_file_path))
