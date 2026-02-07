#!/bin/env python3

import argparse

from hashlib import md5


def part_a(input: str):
    print('Part A')

    password = ''
    index = 0
    running = False
    while running == False:
        next_val = bytes(input + str(index), 'utf-8')
        val = md5(next_val).hexdigest()

        if val[0:5] == '00000':
            password += str(val[5])

        if len(password) == 8:
            break

        index += 1       

    print(password)

def part_b(input: str):
    print('Part B')

    password = [None, None, None, None, None, None, None, None]
    index = 0
    successes = 0
    running = False
    while running == False:
        next_val = bytes(input + str(index), 'utf-8')
        val = md5(next_val).hexdigest()

        start = val[0:5]
        pos = int(val[5], 16)

        if start == '00000' and (0 <= pos) and (7 >= pos):

            if password[pos] == None:
                next_char = val[6]
                password[pos] = next_char
                successes += 1

        if successes == 8:
            break

        index += 1       

    delimiter = ""
    formatted_password = delimiter.join(password)
    print(formatted_password)


if __name__ == "__main__":

    print("Advent of code 2016 - Day 06")

    parser = argparse.ArgumentParser(
        prog='Advent of code 2016 - Day 05',
        description='Runs the solution for AoC 2016 Day 05',
        epilog='Merry Christmas (Feb 2026)'
    )

    parser.add_argument('-i', '--input')
    parser.add_argument('-r', '--run')

    args = parser.parse_args()

    if args.input is None:
        exit(1)

    if args.run is None:
        exit(1)

    if args.run == "part_a":
        part_a(args.input)
    elif args.run == "part_b":
        part_b(args.input)

