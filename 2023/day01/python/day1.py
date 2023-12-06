#!/bin/env python3

import argparse
import os
from functools import reduce

number_words = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def is_number(char):
    if (ord(char) >= 48) and (ord(char) <= 57):
        return True
    return False


def first_last_num(list):
    return [list[0], list[-1]]


def convert_words_to_numbers(input_string):
    curr_string = ""

    for char in input_string:
        curr_string += char
        for number_word, number_str in number_words.items():
            curr_string = curr_string.replace(number_word, number_str)

    return curr_string


def process_file_a(path):
    with open(path) as file:
        total = sum(
            map(
                lambda x: int("".join(first_last_num(list(filter(is_number, x))))),
                file.readlines(),
            )
        )
        return total


def process_file_b(path):
    with open(path) as file:
        total = sum(
            map(
                lambda x: int(
                    "".join(
                        first_last_num(
                            list(filter(is_number, convert_words_to_numbers(x)))
                        )
                    )
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
