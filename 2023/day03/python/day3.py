#!/bin/env python3

import argparse
import os
import re


def part_a(input_file_path):
    with open(input_file_path) as file:
        for line in file.readlines():
            if match := re.search(r"[0-9]{1,}", line):
                pass


def part_b(input_file_path):
    with open(input_file_path) as file:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    print(part_a(args.input_file_path))
    print(part_b(args.input_file_path))
