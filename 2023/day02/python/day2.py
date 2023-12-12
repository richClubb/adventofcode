#!/bin/env python3
import argparse
import os
import re
from functools import reduce


def get_red_cubes_count(input_round):
    if match := re.search("([0-9]{1,})(red)", input_round):
        return int(match.group(1))
    return 0


def get_green_cubes_count(input_round):
    if match := re.search("([0-9]{1,})(green)", input_round):
        return int(match.group(1))
    return 0


def get_blue_cubes_count(input_round):
    if match := re.search("([0-9]{1,})(blue)", input_round):
        return int(match.group(1))
    return 0


def cubes_in_round(input_round):
    return (
        get_red_cubes_count(input_round),
        get_green_cubes_count(input_round),
        get_blue_cubes_count(input_round),
    )


def max_cubes_in_game(input_game):
    return reduce(
        lambda x, y: (max(x[0], y[0]), max(x[1], y[1]), max(x[2], y[2])),
        map(cubes_in_round, input_game.split(";")),
    )


def check_red(input_round):
    if match := re.search("([0-9]{1,})(red)", input_round):
        if int(match.group(1)) > 12:
            return False
    return True


def check_green(input_round):
    if match := re.search("([0-9]{1,})(green)", input_round):
        if int(match.group(1)) > 13:
            return False
    return True


def check_blue(input_round):
    if match := re.search("([0-9]{1,})(blue)", input_round):
        if int(match.group(1)) > 14:
            return False
    return True


def is_round_possible(input_round):
    if (
        len(
            list(
                filter(
                    lambda x: (check_red(x) == False)
                    or (check_green(x) == False or (check_blue(x) == False)),
                    input_round.split(","),
                )
            )
        )
        > 0
    ):
        return False
    return True


def is_game_possible(input_game_string):
    if (
        len(
            list(
                filter(
                    lambda x: is_round_possible(x) == False,
                    input_game_string.split(";"),
                )
            )
        )
        > 0
    ):
        return False
    return True


def part_a(input_file_path):
    with open(input_file_path) as file:
        return sum(
            map(
                lambda x: int(x[0]) if is_game_possible(x[1]) else 0,
                map(
                    lambda y: y.strip().replace(" ", "").replace("Game", "").split(":"),
                    file.readlines(),
                ),
            )
        )


def part_b(input_file_path):
    with open(input_file_path) as file:
        return reduce(
            lambda x, y: x + y[0] * y[1] * y[2],
            map(
                max_cubes_in_game,
                map(
                    lambda x: x.strip()
                    .replace(" ", "")
                    .replace("Game", "")
                    .split(":")[1],
                    file.readlines(),
                ),
            ),
            0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    print(part_a(args.input_file_path))
    print(part_b(args.input_file_path))
