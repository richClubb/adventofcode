#!/bin/env python3


def correct_seed_range(seed_range):
    if seed_range is not None:
        seed_range_start, seed_range_end = seed_range
        if seed_range_start == seed_range_end:
            return (seed_range_start, None)
        return (seed_range_start, seed_range_end)
    return None


def correct_seed_ranges(seed_ranges):
    if seed_ranges is not None:
        temp_output = []
        for seed_range in seed_ranges:
            temp_output.append(correct_seed_range(seed_range))
        return temp_output
    return None


def calculate_new_seeds(seed_range, mappings):
    mapping_output = None
    remaining_seed_range = None
    working_seed_start, working_seed_end = seed_range
    if working_seed_end is None:
        for mapping_dest_start, mapping_src_start, mapping_range in mappings:
            mapping_src_end = mapping_src_start + mapping_range - 1

            if working_seed_start < mapping_src_start:
                mapping_output = [(working_seed_start, None)]

            elif (working_seed_start >= mapping_src_start) and (
                working_seed_start <= mapping_src_end
            ):
                offset = working_seed_start - mapping_src_start
                mapped_value = mapping_dest_start + offset
                mapping_output = [(mapped_value, mapped_value)]
            elif working_seed_start > mapping_src_end:
                pass
            else:
                raise Exception("Invalid range in single value processing")

    else:
        mapping_output = None
        remaining_seed_range = None

        for mapping_dest_start, mapping_src_start, mapping_range in mappings:
            mapping_src_end = mapping_src_start + mapping_range - 1
            mapping_dest_end = mapping_dest_start + mapping_range - 1

            if working_seed_end < mapping_src_start:
                pass
            elif working_seed_start > mapping_src_end:
                pass
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end == mapping_src_start
            ):
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_start)
                mapping_output = [lower_range, mapped_range]
                break
            elif (
                (working_seed_start < mapping_src_start)
                and (working_seed_end > mapping_src_start)
                and (working_seed_end < mapping_src_end)
            ):
                lower_range = (working_seed_start, mapping_src_start - 1)
                offset = working_seed_end - mapping_src_start
                mapped_range = (mapping_dest_start, mapping_dest_start + offset)
                mapping_output = [lower_range, mapped_range]
                break
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [lower_range, mapped_range]
                break
            elif (working_seed_start < mapping_src_start) and (
                working_seed_end > mapping_src_end
            ):
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [lower_range, mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end < mapping_src_end
            ):
                offset = working_seed_end - mapping_src_start
                mapped_range = (mapping_dest_start, mapping_dest_start + offset)
                mapping_output = [mapped_range]
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [mapped_range]
                break
            elif (working_seed_start == mapping_src_start) and (
                working_seed_end > mapping_src_end
            ):
                mapped_range = (mapping_dest_start, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start > mapping_src_start) and (
                working_seed_end < mapping_src_end
            ):
                offset = working_seed_start - mapping_src_start
                size = working_seed_end - working_seed_start
                mapped_range = (
                    mapping_dest_start + offset,
                    mapping_dest_start + offset + size,
                )
                mapping_output = [mapped_range]
                break
            elif (working_seed_start > mapping_src_start) and (
                working_seed_end == mapping_src_end
            ):
                offset = working_seed_start - mapping_src_start
                mapped_range = (mapping_dest_start + offset, mapping_dest_end)
                mapping_output = [mapped_range]
                break
            elif (
                (working_seed_start > mapping_src_start)
                and (working_seed_start < mapping_src_end)
                and (working_seed_end > mapping_src_end)
            ):
                offset = working_seed_start - mapping_src_start
                mapped_range = (mapping_dest_start + offset, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (mapping_src_end + 1, working_seed_end)
                break
            elif (working_seed_start == mapping_src_end) and (
                working_seed_end > mapping_src_end
            ):
                mapped_range = (mapping_dest_end, mapping_dest_end)
                mapping_output = [mapped_range]
                remaining_seed_range = (working_seed_start + 1, working_seed_end)
                break
            else:
                raise Exception("Error case in ranges processing")

    if (mapping_output is None) and (remaining_seed_range is None):
        return [seed_range], None

    mapping_output = correct_seed_ranges(mapping_output)
    remaining_seed_range = correct_seed_range(remaining_seed_range)

    return mapping_output, remaining_seed_range
