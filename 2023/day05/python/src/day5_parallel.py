#!/bin/env python3

from collections import defaultdict
import argparse
import os
from enum import Enum
from multiprocessing import Pool

import functools
import math

RUNS = [
    "part_a", 
    "part_b_forward", 
    "part_b_forward_inline", 
    "part_b_forward_inline_sorted", 
    "part_b_forward_parallel", 
    "part_b_inverse", 
    "part_b_inverse_parallel"
]


class Mapping_Direction(Enum):
    INPUT_TO_OUTPUT = 1
    OUTPUT_TO_INPUT = 2

def do_mapping_forward(input, dest, src, length):
    if input >= src and input < src + length:
        return dest + (input - src)
    # implicit return None

def do_mapping_inverse(input, dest, src, length):
    if input >= dest and input < dest + length:
        return src + (input - dest)
    # implicit return None

def find_location_forward(seed, maps):

    map_key = 0
    while map_key < 7:
        for entry in maps[map_key]:
            x = do_mapping_forward(seed, *entry)
            if x is not None:
                seed = x
                break
        map_key += 1

    return seed

def find_location_forward_inline(seed, maps):

    map_key = 0
    while map_key < 7:
        for entry in maps[map_key]:
            if seed >= entry[1] and seed < entry[1] + entry[2]:
                seed = entry[0] + (seed - entry[1])
                break
        map_key += 1

    return seed

def find_min_in_range(start, size, maps):

    min_loc = 10**30
    for seed in range(start, start + size):
        map_key = 0
        while map_key < 7:
            for entry in maps[map_key]:
                if seed >= entry[1] and seed < entry[1] + entry[2]:
                    seed = entry[0] + (seed - entry[1])
                    break
            if seed < min_loc:
                min_loc = seed
            map_key += 1

    return min_loc

def find_location_inverse(seed, maps):
    
    map_key = 6
    inter = seed
    while map_key >= 0:
        for entry in maps[map_key]:
            x = do_mapping_inverse(inter, *entry)
            if x is not None:
                inter = x
                break
        map_key -= 1

    return inter
    

def find_lowest_location(arguments):
    seed_start, seed_length = arguments[0]
    maps = arguments[1]

    min_loc = 10**30

    for seed in range(seed_start, seed_start + seed_length):
        loc = find_location_forward_inline(seed, maps)
        if loc < min_loc:
            min_loc = loc

    print("{} {} {}".format(seed_start, seed_length, min_loc))
    return min_loc


def find_lowest_seed(arguments):
    location_start, location_end = arguments[0]
    seed_pairs = arguments[1]
    maps = arguments[2]

    for location in range(location_start, location_end):
        calculated_seed = find_location_inverse(
            location, maps
        )

        for start, length in seed_pairs:
            if (calculated_seed >= start) and (calculated_seed <= start + length):
                return location

    return None


def extract_maps_and_seeds(input_file_path, sort=False):
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

    if sort:
        for index in range(0, 7):
            maps[index] = sorted(maps[index], key=lambda entry: entry[1])

    return maps, seeds


def part_a(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    min_loc = 10**30
    for seed in seeds:
        loc = find_location_forward(seed, maps)
        if loc < min_loc:
            min_loc = loc

    return min_loc


def split_range(start, size, ideal_size):
    if size <= ideal_size:
        new_list = [(start,size)]
        return new_list
    
    if size / ideal_size <= 2:
        new_size = size / 2
        new_list = [(start, new_size), (start+new_size, new_size)]
        return new_list

    if size / ideal_size > 2:
        number = int(math.floor(size / ideal_size))
        new_size = int(math.ceil(size / number))

        new_list = []
        remaining = size
        new_start = start
        while remaining > 0:
            if remaining > new_size:
                new_list.append((new_start, new_size))
                new_start += new_size
                remaining -= new_size
            else:
                new_list.append((new_start, remaining))
                remaining -= remaining

        return new_list
    pass


def part_b_forward_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)
    pool_arguments = []

    seed_pairs = sorted(seed_pairs, key=lambda entry: entry[1])

    processors = 28

    ideal_size = round(functools.reduce(lambda x, entry: x + entry[1], seed_pairs, 0) / processors)
    print(ideal_size)

    new_input = []
    for entry in seed_pairs:
        new_input += (split_range(entry[0], entry[1], ideal_size))

    print(len(new_input), new_input)
    for seed_pair in new_input:
        pool_arguments.append((seed_pair, maps))

    with Pool(processors) as p:
        results = p.map(find_lowest_location, pool_arguments)

    return min(results)


def part_b_inverse_multiprocess(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)
    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    range_start = 0
    processes = 28
    range_size = 100000
    while True:
        pool_arguments = []

        for _ in range(0, processes):
            pool_arguments.append(
                ((range_start, range_start + range_size - 1), seed_pairs, maps)
            )
            range_start += range_size

        with Pool(28) as p:
            results = p.map(find_lowest_seed, pool_arguments)

            results = list(filter(lambda x: x is not None, results))

            if len(results) > 0:
                results.sort()
                return results[0]


def part_b_forward(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    min_loc = 10**30
    for seed_start, seed_size in seed_pairs:
        for seed in range(seed_start, seed_start + seed_size - 1):
            loc = find_location_forward(seed, maps)
            if loc < min_loc:
                min_loc = loc

    return min_loc

def part_b_forward_inline(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    min_loc = 10**30
    for seed_start, seed_size in seed_pairs:
        range_result = find_min_in_range(seed_start, seed_size, maps)
        if range_result < min_loc:
            min_loc = range_result

    return min_loc

def part_b_forward_inline_sorted(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path, True)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    min_loc = 10**30
    for seed_start, seed_size in seed_pairs:
        for seed in range(seed_start, seed_start + seed_size - 1):
            loc = find_location_forward_inline(seed, maps)
            if loc < min_loc:
                min_loc = loc

    return min_loc


def part_b_inverse(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    start_val = 1
    while True:
        try:
            calculated_seed = find_location_inverse(
                start_val, maps
            )

            for start, length in seed_pairs:
                if (calculated_seed >= start) and (calculated_seed <= start + length):
                    return start_val

            start_val += 1

        except KeyboardInterrupt:
            print(f"exited on {start_val}")
            exit()

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
            f"part b (forward depth first single process): {part_b_forward(args.input_file_path)}"
        )
    elif args.run == "part_b_forward_inline":
        print(
            f"part b (forward depth first single process): {part_b_forward_inline(args.input_file_path)}"
        )
    elif args.run == "part_b_forward_inline_sorted":
        print(
            f"part b (forward depth first single process): {part_b_forward_inline_sorted(args.input_file_path)}"
        )
    elif args.run == "part_b_forward_parallel":
        print(
            f"part b (forward depth first multiprocess): {part_b_forward_multiprocess(args.input_file_path)}"
        )
    elif args.run == "part_b_inverse":
        print(
            f"part b (inverse depth first): {part_b_inverse(args.input_file_path)}"
        )
    elif args.run == "part_b_inverse_parallel":
        print(
            f"part b (inverse depth first multiprocess): {part_b_inverse_multiprocess(args.input_file_path)}"
        )
