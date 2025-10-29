#!/bin/env python3

from collections import defaultdict
import argparse
import os
from enum import Enum
from multiprocessing import Pool

RUNS = ["part_a", "part_b_ranges"]


class Mapping_Direction(Enum):
    INPUT_TO_OUTPUT = 1
    OUTPUT_TO_INPUT = 2


def case_1(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed below the mapping range
    """
    if seed_end < mapping_src_start:
        return None, None
    return case_2(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_2(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed above the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    if seed_start > mapping_src_end:
        return None, None
    return case_3(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_3(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed just overlapping with the first value of the mapping range
    """
    if (seed_start < mapping_src_start) and (seed_end == mapping_src_start):
        lower_range = (seed_start, mapping_src_start - 1)
        mapped_range = (mapping_dst_start, mapping_dst_start)
        return [lower_range, mapped_range], None
    return case_4(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_4(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed overlapping with the beginning of the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    if (
        (seed_start < mapping_src_start)
        and (seed_end > mapping_src_start)
        and (seed_end < mapping_src_end)
    ):
        # Case 4
        lower_range = (seed_start, mapping_src_start - 1)
        offset = seed_end - mapping_src_start
        mapped_range = (mapping_dst_start, mapping_dst_start + offset)
        return [lower_range, mapped_range], None
    return case_5(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_5(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed overflowing the beginning of the mapping range but fully encompassed in
    the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start < mapping_src_start) and (seed_end == mapping_src_end):
        lower_range = (seed_start, mapping_src_start - 1)
        mapped_range = (mapping_dst_start, mapping_dst_end)
        return [lower_range, mapped_range], None
    return case_6(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_6(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed overflowing the start and end of the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start < mapping_src_start) and (seed_end > mapping_src_start):
        lower_range = (seed_start, mapping_src_start - 1)
        mapped_range = (mapping_dst_start, mapping_dst_end)
        remaining_seed_range = (mapping_src_end + 1, seed_end)
        return [lower_range, mapped_range], remaining_seed_range
    return case_7(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_7(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed starting at the beginning of the mapping range but contained inside
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    if (seed_start == mapping_src_start) and (seed_end < mapping_src_end):
        offset = seed_end - mapping_src_start
        mapped_range = (mapping_dst_start, mapping_dst_start + offset)
        return [mapped_range], None
    return case_8(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_8(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed matches the entire mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start == mapping_src_start) and (seed_end == mapping_src_end):
        mapped_range = (mapping_dst_start, mapping_dst_end)
        return [mapped_range], None
    return case_9(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_9(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed matching the beginning of the mapping range but overflowing the end
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start == mapping_src_start) and (seed_end > mapping_src_end):
        mapped_range = (mapping_dst_start, mapping_dst_end)
        return [mapped_range], (mapping_src_end + 1, seed_end)
    return case_10(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_10(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed starts and ends in the middle of the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    if (seed_start > mapping_src_start) and (seed_end < mapping_src_end):
        offset = seed_start - mapping_src_start
        size = seed_end - seed_start
        mapped_range = (
            mapping_dst_start + offset,
            mapping_dst_start + offset + size,
        )
        return [mapped_range], None
    return case_11(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_11(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed starts in the middle of the mapping range and ends at the end.
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start > mapping_src_start) and (seed_end == mapping_src_end):
        offset = seed_start - mapping_src_start
        mapped_range = (mapping_dst_start + offset, mapping_dst_end)
        return [mapped_range], None
    return case_12(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_12(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed starts in the middle of the mapping range but overflows the end
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (
        (seed_start > mapping_src_start)
        and (seed_start < mapping_src_end)
        and (seed_end > mapping_src_end)
    ):
        # Case 12
        offset = seed_start - mapping_src_start
        mapped_range = (mapping_dst_start + offset, mapping_dst_end)
        return [mapped_range], (mapping_src_end + 1, seed_end)
    return case_13(
        seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size
    )


def case_13(seed_start, seed_end, mapping_src_start, mapping_dst_start, mapping_size):
    """
    Seed overlaps with the last value of the mapping range
    """
    mapping_src_end = mapping_src_start + mapping_size - 1
    mapping_dst_end = mapping_dst_start + mapping_size - 1
    if (seed_start == mapping_src_end) and (seed_end > mapping_src_end):
        # Case 13
        mapped_range = (mapping_dst_end, mapping_dst_end)
        return [mapped_range], (seed_start + 1, seed_end)
    return None


def clean_up_seed_list(seed_ranges):
    """
    Used to help print things nicely by making the ranges continuous, not necessary
    but nice.
    """
    output_ranges = []
    starting_range_start, starting_range_end = seed_ranges.pop(0)
    while len(seed_ranges) > 0:
        working_range_start, working_range_end = seed_ranges.pop(0)

        if (starting_range_end is None) and (working_range_end is None):
            if starting_range_start == working_range_start:
                pass
            elif starting_range_start + 1 == working_range_start:
                starting_range_end = working_range_start
            else:
                output_ranges.append((starting_range_start, None))
                starting_range_start = working_range_start
        elif (starting_range_end is None) and (working_range_end is not None):
            if starting_range_start == working_range_start + 1:
                starting_range_end == working_range_end
            else:
                output_ranges.append((starting_range_start, None))
                starting_range_start, starting_range_end = (
                    working_range_start,
                    working_range_end,
                )
        elif (starting_range_end is not None) and (working_range_end is None):
            if (starting_range_end + 1) == working_range_start:
                starting_range_end = working_range_start
            else:
                output_ranges.append((starting_range_start, starting_range_end))
                starting_range_start, starting_range_end = (
                    working_range_start,
                    working_range_end,
                )
        else:
            if (starting_range_end + 1) == working_range_start:
                starting_range_end = working_range_end
            elif starting_range_end == working_range_end:
                starting_range_end = working_range_end
            elif starting_range_end > working_range_end:
                pass
            elif (starting_range_end >= working_range_start) and (
                starting_range_start <= working_range_end
            ):
                starting_range_end = working_range_end
            else:
                output_ranges.append((starting_range_start, starting_range_end))
                starting_range_start, starting_range_end = (
                    working_range_start,
                    working_range_end,
                )

    output_ranges.append((starting_range_start, starting_range_end))
    return output_ranges


def correct_seed_range(seed_range):
    """
    If the seed range is a single value then correct it
    Check for any erroneous values.
    """
    if seed_range is not None:
        seed_range_start, seed_range_end = seed_range
        if seed_range_start == seed_range_end:
            return (seed_range_start, None)
        elif (seed_range_end is not None) and (seed_range_start > seed_range_end):
            raise Exception(f"Seed range invalid: {seed_range_start}, {seed_range_end}")
        return (seed_range_start, seed_range_end)
    return None


def correct_seed_ranges(seed_ranges):
    """ """
    if seed_ranges is not None:
        temp_output = []
        for seed_range in seed_ranges:
            temp_output.append(correct_seed_range(seed_range))
        return temp_output
    return None


def find_map(seed, mappings):
    """
    Finds the right map for mapping
    """
    for map in mappings:
        _, map_src_start, map_size = map
        map_src_end = map_src_start + map_size - 1
        if seed >= map_src_start and seed <= map_src_end:
            return map


def map_seed(seed, mappings):
    if map := find_map(seed, mappings):
        map_dest_start, map_src_start, _ = map
        mapped_value = map_dest_start + seed - map_src_start
        return [(mapped_value, mapped_value)]
    else:
        return [(seed, None)]


def calculate_new_seeds(seed_range, mappings):
    """
    calculates the new seed ranges.
    """
    working_seed_start, working_seed_end = seed_range
    map_list = sorted(mappings, key=lambda lst: lst[1])
    if working_seed_end is None:
        mapping_output = map_seed(working_seed_start, mappings)
        remaining_seed_range = None

    else:
        for mapping_dest_start, mapping_src_start, mapping_range in map_list:
            if result := case_1(
                working_seed_start,
                working_seed_end,
                mapping_src_start,
                mapping_dest_start,
                mapping_range,
            ):
                mapping_output = result[0]
                remaining_seed_range = result[1]
            else:
                raise Exception("Error case in ranges processing")

            if (mapping_output is not None) or (remaining_seed_range is not None):
                break

    if (mapping_output is None) and (remaining_seed_range is None):
        return [seed_range], None

    return correct_seed_ranges(mapping_output), correct_seed_range(remaining_seed_range)


def process_seed_mapping(seed_list, mapping):
    """

    returns: A sorted list of seed pairs
    """
    new_seeds = []
    # Takes each seed pair and calculates the mapping, any remaining not mapped will be
    # re-added to the seed_list
    while len(seed_list) > 0:
        working_seed = seed_list.pop(0)
        mapped_seeds, remaining_seeds = calculate_new_seeds(working_seed, mapping)
        if mapped_seeds is not None:
            new_seeds += mapped_seeds

        # Add any remaining to the seed list
        if remaining_seeds is not None:
            seed_list.append(remaining_seeds)
            seed_list = sorted(seed_list, key=lambda tup: tup[0])

    mapped_seed_pairs = sorted(new_seeds, key=lambda tup: tup[0])
    return mapped_seed_pairs


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

    # gets the seed ranges in a (start, end) format rather than (start, size)
    seed_ranges = []
    for seed in seeds:
        seed_ranges.append((seed, None))

    # The seed ranges have to be sorted for this algorithm to work
    seed_ranges = sorted(seed_ranges, key=lambda tup: tup[0])

    # Calculates the new seed ranges for each layer
    for map_index in range(0, len(maps.keys())):
        curr_map_list = maps[map_index]
        seed_ranges = process_seed_mapping(seed_ranges, curr_map_list)

    return seed_ranges[0][0]


def part_b_forward_ranges(input_file_path):
    maps, seeds = extract_maps_and_seeds(input_file_path)

    # Extracts a list of pairs (start, size)
    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds, 2)

    # gets the seed ranges in a (start, end) format rather than (start, size)
    seed_ranges = []
    for seed_start, seed_range_size in seed_pairs:
        seed_ranges.append((seed_start, seed_start + seed_range_size - 1))

    # The seed ranges have to be sorted for this algorithm to work
    seed_ranges = sorted(seed_ranges, key=lambda tup: tup[0])

    # Calculates the new seed ranges for each layer
    for map_index in range(0, len(maps.keys())):
        curr_map_list = maps[map_index]
        seed_ranges = process_seed_mapping(seed_ranges, curr_map_list)

    return seed_ranges[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file_path")
    parser.add_argument("run")

    args = parser.parse_args()

    if os.path.exists(args.input_file_path) is False:
        print("Missing input file")
        exit()

    if args.run == "part_a":
        print(f"part a (forward depth first): {part_a(args.input_file_path)}")
    elif args.run == "part_b_ranges":
        print(
            f"part b (forward ranges method): {part_b_forward_ranges(args.input_file_path)}"
        )
