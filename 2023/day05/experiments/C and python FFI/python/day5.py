#!/bin/env python3

from ctypes import *
import argparse
import os
from multiprocessing import Pool

seed_range_lib = cdll.LoadLibrary("../c_lib_src/lib/release/libseed_calc.so")

class SeedRange(Structure):
    _fields_ = [
        ("start", c_uint64),
        ("size", c_uint64)
    ]

class SeedMap(Structure):
    _fields_ = [
        ("source", c_uint64),
        ("target", c_uint64),
        ("size", c_uint64)
    ]

class SeedMapLayer(Structure):
    _fields_ = [
        ("num_seed_maps", c_uint64),
        ("seed_maps", POINTER(SeedMap))
    ]

class SeedMapLayers(Structure):
    _fields_ = [
        ("num_seed_map_layers", c_uint64),
        ("seed_map_layers", POINTER(SeedMapLayer))
    ]

def injest_file(input_file_path):
    seed_maps_raw = []

    with open(input_file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seeds: "):
                seeds = list(map(int, line.split()[1:]))
                continue
            elif line.startswith("seed-to-soil"):
                seed_maps_raw.append([])
                active_map = 0
            elif line.startswith("soil-to-fertilizer"):
                seed_maps_raw.append([])
                active_map = 1
            elif line.startswith("fertilizer-to-water"):
                seed_maps_raw.append([])
                active_map = 2
            elif line.startswith("water-to-light"):
                seed_maps_raw.append([])
                active_map = 3
            elif line.startswith("light-to-temperature"):
                seed_maps_raw.append([])
                active_map = 4
            elif line.startswith("temperature-to-humidity"):
                seed_maps_raw.append([])
                active_map = 5
            elif line.startswith("humidity-to-location"):
                seed_maps_raw.append([])
                active_map = 6
            elif line:
                values = list(map(int, line.split()))
                seed_maps_raw[active_map].append(SeedMap(source=c_uint64(values[1]), target=c_uint64(values[0]), size=c_uint64(values[2])))

    seed_map_layers_raw = []
    for seed_maps in seed_maps_raw:
        seed_map_layers_raw.append(SeedMapLayer(num_seed_maps=len(seed_maps), seed_maps=(SeedMap * len(seed_maps))(*seed_maps)))

    return seeds, SeedMapLayers(num_seed_map_layers=len(seed_map_layers_raw), seed_map_layers=(SeedMapLayer * len(seed_map_layers_raw))(*seed_map_layers_raw))

def extract_maps_and_seeds(input_file_path):
    seeds, maps = injest_file(input_file_path)

    seeds = list(map(c_uint64, seeds))    
    return seeds, maps

def extract_maps_and_seed_ranges(input_file_path):
    seeds, maps = injest_file(input_file_path)

    seed_ranges_raw = []
    for index in range(0, len(seeds), 2):
        seed_ranges_raw.append(SeedRange(start=seeds[index], size=seeds[index+1]))

    return seed_ranges_raw, maps


def part_a(input_file_path):
    seeds, maps = extract_maps_and_seeds(input_file_path)

    seed_map_layers_map_seed = seed_range_lib.seed_map_layers_map_seed
    # seed_map_layers_map_seed.argtypes = (
    #     c_uint64, POINTER(SeedMapLayers)
    # )
    seed_map_layers_map_seed.restype = c_uint64

    min_value = 18446744073709551615
    for seed in seeds:
        value = seed_map_layers_map_seed(seed, byref(maps))
        if value < min_value:
            min_value = value

    return min_value

def part_b(input_file_path):
    seed_ranges, maps = extract_maps_and_seed_ranges(input_file_path)

    seed_map_layers_map_seed = seed_range_lib.seed_map_layers_map_seed
    seed_map_layers_map_seed.restype = c_uint64

    seed_range_find_min = seed_range_lib.seed_range_find_min
    seed_range_find_min.restype = c_int64

    min_value = 18446744073709551615
    for seed_range in seed_ranges:
        value = seed_range_find_min(byref(seed_range), byref(maps))
        if value < min_value:
            min_value = value

    return min_value

RUNS = ["part_a", "part_b"]

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
    elif args.run == "part_b":
        print(f"part b (forward depth first): {part_b(args.input_file_path)}")
