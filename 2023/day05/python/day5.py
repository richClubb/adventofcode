#!/bin/env python3

from collections import defaultdict
import argparse
import os
from enum import Enum
from multiprocessing import Pool


class Mapping_Direction(Enum):
    INPUT_TO_OUTPUT = 1
    OUTPUT_TO_INPUT = 2


def calculate_translation_ranges(map):
    pass


def do_mapping(input, dest, src, length, direction=Mapping_Direction.INPUT_TO_OUTPUT):
    if direction == Mapping_Direction.INPUT_TO_OUTPUT:
        if input >= src and input < src + length:
            return dest + (input - src)
    elif direction == Mapping_Direction.OUTPUT_TO_INPUT:
        if input >= dest and input < dest + length:
            return src + (input - dest)
    # implicit return None


def find_location_wrapper(arguments):
    seed_start = arguments[0]
    seed_length = arguments[1]
    maps = arguments[2]
    min_loc = 10**30
    for seed in range(seed_start, seed_start + seed_length):
        loc = find_location(seed, maps)
        if loc < min_loc:
            min_loc = loc
    return min_loc


def find_location(seed, maps, direction=Mapping_Direction.INPUT_TO_OUTPUT):
    if direction == Mapping_Direction.OUTPUT_TO_INPUT:
        map_key = 6
        inter = seed
        while map_key >= 0:
            for entry in maps[map_key]:
                x = do_mapping(inter, *entry, direction=direction)
                if x is not None:
                    inter = x
                    break
            map_key -= 1

        return inter

    elif direction == Mapping_Direction.INPUT_TO_OUTPUT:
        map_key = 0
        while map_key < 7:
            for entry in maps[map_key]:
                x = do_mapping(seed, *entry, direction=direction)
                if x is not None:
                    seed = x
                    break
            map_key += 1

        return seed


def extract_maps_and_seeds(input_file_path):
    maps = defaultdict(list)

    with open(input_file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seeds"):
                seeds = list(map(int, line.split()[1:]))
            elif line.startswith("seed-to-soil"):
                active_map = 0
            elif line.startswith("soil-to-fertilizer"):
                active_map = 1
            elif line.startswith("fertilizer-to-water"):
                active_map = 2
            elif line.startswith("water-to-light"):
                active_map = 3
            elif line.startswith("light-to-temperature"):
                active_map = 4
            elif line.startswith("temperature-to-humidity"):
                active_map = 5
            elif line.startswith("humidity-to-location"):
                active_map = 6
            elif line:
                maps[active_map].append(list(map(int, line.split())))

    return maps, seeds


def part_a(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    min_loc = 10**30
    for seed in seeds:
        loc = find_location(seed, maps)
        if loc < min_loc:
            min_loc = loc

    return min_loc


def part_b_backward(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    max_seed = 0
    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    for start, length in seed_pairs:
        max_seed = max(max_seed, start + length)

    start_val = 0
    while True:
        try:
            calculated_seed = find_location(
                start_val, maps, direction=Mapping_Direction.OUTPUT_TO_INPUT
            )

            for start, length in seed_pairs:
                if (calculated_seed >= start) and (calculated_seed <= start + length):
                    return start_val

            start_val += 1

        except KeyboardInterrupt:
            print(f"exited on {start_val:,}")
            exit()


def part_b_forward_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    arguments = []

    for seed_pair in seed_pairs:
        arguments.append(seed_pair + [maps])

    with Pool(6) as p:
        results = p.map(find_location_wrapper, arguments)

    return min(results)


def part_b_forward_ranges(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    arguments = []

    for seed_pair in seed_pairs:
        arguments.append(seed_pair + [maps])

    with Pool(6) as p:
        results = p.map(find_location_wrapper, arguments)

    return min(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    # print(f"part a: {part_a(args.input_file_path)}")
    # print(f"part b backwards: {part_b_backward(args.input_file_path)}")
    # print(f"part b forwards multiprocess (bad): {part_b_forward_multiprocess(args.input_file_path)}")
    # print(f"part b forwards ranges (unknown): {part_b_forward_ranges(args.input_file_path)}")
