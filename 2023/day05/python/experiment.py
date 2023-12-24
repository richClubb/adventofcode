#!/bin/env python3


def case_3(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_4(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_5(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_6(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_7(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_8(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_9(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_10(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_11(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_12(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


def case_13(seed_start, seed_end, mapping_src_start, mapping_dst_start):
    return None, None


check_functions = [
    case_3,
    case_4,
    case_5,
    case_6,
    case_7,
    case_8,
    case_9,
    case_10,
    case_11,
    case_12,
    case_13,
]


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
    if seed_range is not None:
        seed_range_start, seed_range_end = seed_range
        if seed_range_start == seed_range_end:
            return (seed_range_start, None)
        elif (seed_range_end is not None) and (seed_range_start > seed_range_end):
            raise Exception(f"Seed range invalid: {seed_range_start}, {seed_range_end}")
        return (seed_range_start, seed_range_end)
    return None


def correct_seed_ranges(seed_ranges):
    if seed_ranges is not None:
        temp_output = []
        for seed_range in seed_ranges:
            temp_output.append(correct_seed_range(seed_range))
        return temp_output
    return None


def find_map(seed, mappings):
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
    working_seed_start, working_seed_end = seed_range
    map_list = sorted(mappings, key=lambda lst: lst[1])
    if working_seed_end is None:
        mapping_output = map_seed(working_seed_start, mappings)
        remaining_seed_range = None

    else:
        for mapping_dest_start, mapping_src_start, mapping_range in map_list:
            mapping_src_end = mapping_src_start + mapping_range - 1
            mapping_dest_end = mapping_dest_start + mapping_range - 1

            if working_seed_end < mapping_src_start:
                # Case 1
                mapping_output = None
                remaining_seed_range = None
            elif working_seed_start > mapping_src_end:
                # Case 2
                mapping_output = None
                remaining_seed_range = None
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end == mapping_src_start
            ):
                # Case 3
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_start)
                mapping_output = [lower_range, mapped_range]
                remaining_seed_range = None
                break
            elif (
                (working_seed_start < mapping_src_start)
                and (working_seed_end > mapping_src_start)
                and (working_seed_end < mapping_src_end)
            ):
                # Case 4
                lower_range = (working_seed_start, mapping_src_start - 1)
                offset = working_seed_end - mapping_src_start
                mapped_range = (mapping_dest_start, mapping_dest_start + offset)
                mapping_output = [lower_range, mapped_range]
                remaining_seed_range = None
                break
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                # Case 5
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [lower_range, mapped_range]
                remaining_seed_range = None
                break
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end > mapping_src_end
            ):
                # Case 6
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [lower_range, mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end < mapping_src_end
            ):
                # Case 7
                offset = working_seed_end - mapping_src_start
                mapped_range = (mapping_dest_start, mapping_dest_start + offset)
                mapping_output = [mapped_range]
                remaining_seed_range = None
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                # Case 8
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = None
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end > mapping_src_end
            ):
                # Case 9
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start > mapping_src_start) and (
                working_seed_end < mapping_src_end
            ):
                # Case 10
                offset = working_seed_start - mapping_src_start
                size = working_seed_end - working_seed_start
                mapped_range = (
                    mapping_dest_start + offset,
                    mapping_dest_start + offset + size,
                )
                mapping_output = [mapped_range]
                remaining_seed_range = None
                break
            elif (working_seed_start > mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                # Case 11
                offset = working_seed_start - mapping_src_start
                mapped_range = (mapping_dest_start + offset, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = None
                break
            elif (
                (working_seed_start > mapping_src_start)
                and (working_seed_start < mapping_src_end)
                and (working_seed_end > mapping_src_end)
            ):
                # Case 12
                offset = working_seed_start - mapping_src_start
                mapped_range = (mapping_dest_start + offset, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start == mapping_src_end) and (
                working_seed_end > mapping_src_end
            ):
                # Case 13
                mapped_range = (mapping_dest_end, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (working_seed_start + 1, working_seed_end)
                break
            else:
                raise Exception("Error case in ranges processing")

    if (mapping_output is None) and (remaining_seed_range is None):
        return [seed_range], None

    return correct_seed_ranges(mapping_output), correct_seed_range(remaining_seed_range)
