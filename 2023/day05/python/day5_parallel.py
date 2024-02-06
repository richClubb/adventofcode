#!/bin/env python3

from collections import defaultdict
import argparse
import os
from enum import Enum
from multiprocessing import Pool
from time import time


RUNS = ["part_a", "part_b_forward", "part_b_inverse"]


class Mapping_Direction(Enum):
    INPUT_TO_OUTPUT = 1
    OUTPUT_TO_INPUT = 2


def do_mapping(input, dest, src, length, direction=Mapping_Direction.INPUT_TO_OUTPUT):
    if direction == Mapping_Direction.INPUT_TO_OUTPUT:
        if input >= src and input < src + length:
            return dest + (input - src)
    elif direction == Mapping_Direction.OUTPUT_TO_INPUT:
        if input >= dest and input < dest + length:
            return src + (input - dest)
    # implicit return None


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


def find_lowest_location(arguments):
    seed_start, seed_length = arguments[0]
    maps = arguments[1]

    min_loc = 10**30

    for seed in range(seed_start, seed_start + seed_length):
        loc = find_location(seed, maps)
        if loc < min_loc:
            min_loc = loc

    return min_loc


def find_lowest_seed(arguments):
    location_start, location_end = arguments[0]
    seed_pairs = arguments[1]
    maps = arguments[2]

    for location in range(location_start, location_end):
        calculated_seed = find_location(
            location, maps, direction=Mapping_Direction.OUTPUT_TO_INPUT
        )

        for start, length in seed_pairs:
            if (calculated_seed >= start) and (calculated_seed <= start + length):
                return location

    return None


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


def part_b_forward_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    pool_arguments = []

    for seed_pair in seed_pairs:
        pool_arguments.append((seed_pair, maps))

    with Pool(4) as p:
        results = p.map(find_lowest_location, pool_arguments)

    return min(results)


def part_b_inverse_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    range_start = 0
    processes = 4
    range_size = 100000
    while True:
        pool_arguments = []

        for _ in range(0, processes):
            pool_arguments.append(
                ((range_start, range_start + range_size - 1), seed_pairs, maps)
            )
            range_start += range_size

        with Pool(4) as p:
            results = p.map(find_lowest_seed, pool_arguments)

            results = list(filter(lambda x: x is not None, results))

            if len(results) > 0:
                results.sort()
                return results[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")
    parser.add_argument("run", choices=RUNS)

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    if args.run == "part_a":
        print(f"part a (forward depth first): {part_a(args.input_file_path)}")
    elif args.run == "part_b_forward":
        print(
            f"part b (forward depth first multiprocess): {part_b_forward_multiprocess(args.input_file_path)}"
        )
    elif args.run == "part_b_inverse":
        print(
            f"part b (inverse depth first multiprocess): {part_b_inverse_multiprocess(args.input_file_path)}"
        )
