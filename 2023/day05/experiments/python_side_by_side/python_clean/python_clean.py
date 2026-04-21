#!/bin/env python3

import argparse
import functools
import math
import os

from multiprocessing import Pool


RUNS = [
    "part_a",
    "part_b",
    "part_b_parallel"
]

def parse_seeds(input_file_path: str):
    with open(input_file_path, 'r') as file:
        for line in file:
            if line.startswith("seeds: "):
                seeds = list(map(int, line.split()[1:]))
                return sorted(seeds)
            
    return None


def parse_seed_ranges(input_file_path: str):
    with open(input_file_path, 'r') as file:
        for line in file:
            if line.startswith("seeds: "):
                seeds = list(map(int, line.split()[1:]))
                f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
                seed_pairs = f(seeds, 2)
                return sorted(seed_pairs)
    return None


def parse_maps(input_file_path: str):
    
    layers = None
    with open(input_file_path, 'r') as file:

        curr_layer = []
        for line in file:
            line = line.strip()
            if line.startswith("seeds: "):
                continue;
            if len(line) == 0:
                continue
            if line.find(":") != -1:
                if layers is None:
                    layers = []
                else:
                    # curr_layer = sorted(curr_layer, key=lambda x: x[1])
                    layers.append(curr_layer)
                curr_layer = []
                continue;
            map_list = list(map(int, line.split()))
            curr_layer.append((map_list[0], map_list[1], map_list[2]))
        # curr_layer = sorted(curr_layer, key=lambda x: x[1])
        layers.append(curr_layer)
    
    return layers


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
    
    print("blah")


def rebalance_ranges(seed_ranges, number):
    
    print(len(seed_ranges))
    print(number)
    if len(seed_ranges) >= number:
        return seed_ranges
    
    ideal_size = round(functools.reduce(lambda x, entry: x + entry[1], seed_ranges, 0) / number)
    print(ideal_size)

    new_ranges = []
    for seed_range in seed_ranges:
        new_ranges += (split_range(seed_range[0], seed_range[1], ideal_size))

    return new_ranges


def map_process_seed(seed: int, map: tuple):
    map_source_start = map[1]
    map_dest = map[0]
    map_size = map[2]
    map_source_end = map_source_start + map_size

    if (seed >= map_source_start) and (seed < map_source_end):
        return (seed - map_source_start) + map_dest
    
    return None


def layer_process_seed(seed: int, layer: list):
    
    for map in layer:
        val = map_process_seed(seed, map)
        if val is not None:
            return val
        
    return seed
        

def process_seed(seed: int, layers: list):
    val = seed
    for layer in layers:
        val = layer_process_seed(val, layer)

    return val


def process_seeds(seeds: list, layers: list):
    min = 2**64
    for seed in seeds:
        val = process_seed(seed, layers)
        if val < min:
            min = val
    
    return min


def process_seed_range(seed_range: list, layers: list):
    start = seed_range[0]
    end = start + seed_range[1]
    min_val = 2**64
    for seed in range(start, end):
        val = process_seed(seed, layers)
        if val < min_val:
            min_val = val
    return min_val


def process_seed_ranges(seed_ranges: list, layers: list):
    min_val = 2**64
    for seed_range in seed_ranges:
        val = process_seed_range(seed_range, layers)
        if val < min_val:
            min_val = val
    return min_val

def process_seed_range_wrap(arguments):
    seed_range = arguments[0]
    layers = arguments[1]

    return process_seed_range(seed_range, layers)


def part_a(input_file_path: str):
    seeds = parse_seeds(input_file_path)
    layers = parse_maps(input_file_path)
    
    return process_seeds(seeds, layers)
    

def part_b(input_file_path: str):
    seed_ranges = parse_seed_ranges(input_file_path)
    layers = parse_maps(input_file_path)
    
    return process_seed_ranges(seed_ranges, layers)


def part_b_parallel(input_file_path: str):
    seed_ranges = parse_seed_ranges(input_file_path)
    layers = parse_maps(input_file_path)

    pool_arguments = []

    processors = 10
    seed_ranges = rebalance_ranges(seed_ranges, processors)

    for seed_range in seed_ranges:
        pool_arguments.append((seed_range, layers))

    with Pool(processors) as p:
        results = p.map(process_seed_range_wrap, pool_arguments)

    return min(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")
    parser.add_argument("run", choices=RUNS)

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    if args.run == "part_a":
        print(part_a(args.input_file_path))
    elif args.run == "part_b":
        print(part_b(args.input_file_path))
    elif args.run == "part_b_parallel":
        print(part_b_parallel(args.input_file_path))
    else:
        print("Unknown run")

    pass