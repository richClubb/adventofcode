#!/bin/env python3

seeds = [
    (127, 17),
    (1, 10),
    (50, 25),
]

seed_mappings = [(11, 1, 10), (30, 48, 5), (40, 55, 7)]


def correct_seed_ranges(seed_ranges):
    if seed_ranges is not None:
        temp_output = []
        for seed_start, seed_end in seed_ranges:
            if seed_start == seed_end:
                temp_output.append(seed_start)
            else:
                temp_output.append((seed_start, seed_end))
        return temp_output
    return None


def calculate_new_seeds(seed_range, mappings):
    if type(seed_range) == int:
        for mapping_dest_start, mapping_src_start, mapping_range in mappings:
            mapping_src_end = mapping_src_start + mapping_range - 1
            mapping_dest_end = mapping_dest_start + mapping_range - 1

            if (seed_range >= mapping_src_start) and (seed_range <= mapping_src_end):
                delta = seed_range - mapping_src_start
                return [(mapping_dest_start + delta)], None

        return seed_range, None

    else:
        working_seed_start, working_seed_end = seed_range
        mapping_output = None
        remaining_seed_range = None

        for mapping_dest_start, mapping_src_start, mapping_range in mappings:
            mapping_src_end = mapping_src_start + mapping_range - 1
            mapping_dest_end = mapping_dest_start + mapping_range - 1

            if (mapping_src_start == working_seed_start) and (
                mapping_src_end == working_seed_end
            ):
                # print(f"Fully matching")
                mapping_output = [(mapping_dest_start, mapping_dest_end)]
                break
            elif (
                (mapping_src_start < working_seed_start)
                and (mapping_src_end > working_seed_start)
                and (mapping_src_end < working_seed_end)
            ):
                # print(f"Overlap at beginning")
                delta = working_seed_start - mapping_src_start
                mapping_output = [(mapping_dest_start + delta, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break
            elif (mapping_src_start < working_seed_start) and (
                mapping_src_end == working_seed_start
            ):
                # print(f"Found overlap on first value of seed")
                mapping_output = [(mapping_dest_end, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break
            elif (mapping_src_start > working_seed_start) and (
                mapping_src_end < working_seed_end
            ):
                # print(f"mapping in the middle")
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapping_output = [lower_range, (mapping_dest_start, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break

            elif mapping_src_end < working_seed_start:
                # print(f"Mapping range too low")
                pass
            elif mapping_src_start > working_seed_end:
                # print(f"Mapping range too high")
                pass
            elif (mapping_src_start > working_seed_start) and (
                mapping_src_end == working_seed_end
            ):
                # print(f"mapping in middle, ends on end seed")
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapping_output = [lower_range, (mapping_dest_start, mapping_dest_end)]
                break
            elif (mapping_src_start < working_seed_start) and (
                mapping_src_end == working_seed_end
            ):
                # print(f"Below base but matching end")
                offset = working_seed_start - mapping_src_start
                mapping_output = [(mapping_dest_start + offset, mapping_dest_end)]
                break
            elif (mapping_src_start < working_seed_start) and (
                mapping_src_end > working_seed_end
            ):
                # print(f"fully encompassed seed")
                start_offset = working_seed_start - mapping_src_start
                end_offset = mapping_src_end - working_seed_end
                mapping_output = [
                    (mapping_dest_start + start_offset, mapping_dest_end - end_offset)
                ]
                break
            elif mapping_src_start == working_seed_end:
                # print(f"Overlap just at the end")
                lower_range = (working_seed_start, working_seed_end - 1)
                mapping_output = [lower_range, (mapping_dest_start, mapping_dest_start)]
                break
            elif (
                (mapping_src_start > working_seed_start)
                and (mapping_src_start < working_seed_end)
                and (mapping_src_end > working_seed_end)
            ):
                # print(f"Start in the middle but ends outside")
                offset = mapping_src_end - working_seed_end
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapping_output = [
                    lower_range,
                    (mapping_dest_start, mapping_dest_end - offset),
                ]
                break
            elif (mapping_src_start == working_seed_start) and (
                mapping_src_end > working_seed_end
            ):
                offset = working_seed_end - working_seed_start
                mapping_output = [(mapping_dest_start, mapping_dest_start + offset)]
                break
            elif (mapping_src_start == working_seed_start) and (
                mapping_src_end < working_seed_end
            ):
                mapping_output = [(mapping_dest_start, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break

    if (mapping_output is None) and (remaining_seed_range is None):
        return [seed_range], None

    mapping_output = correct_seed_ranges(mapping_output)
    remaining_seed_range = correct_seed_ranges(remaining_seed_range)

    return mapping_output, remaining_seed_range


if __name__ == "__main__":
    seed_range = []

    for seed_start, seed_range_size in seeds:
        seed_range.append((seed_start, seed_start + seed_range_size - 1))

    seed_range = sorted(seed_range, key=lambda tup: tup[0])

    mapping = []
    while len(seed_range) > 0:
        working_seed_range = seed_range.pop(0)
        temp_mapping, remaining_seed_range = calculate_new_seeds(
            working_seed_range, seed_mappings
        )

        mapping += temp_mapping
        if remaining_seed_range is not None:
            seed_range += remaining_seed_range
            seed_range = sorted(seed_range, key=lambda tup: tup[0])

        individual_seeds = []
        seed_ranges = []
        for entry in mapping:
            if type(entry) == int:
                individual_seeds.append(entry)
            if type(entry) == tuple:
                seed_ranges.append(entry)

        individual_seeds.sort()
        seed_ranges = sorted(seed_ranges, key=lambda tup: tup[0])
