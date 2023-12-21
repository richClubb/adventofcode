#!/bin/env python3

seeds = [
    (127, 17),
    (1, 10),
    (50, 25),
]

seed_mappings = [(11, 1, 10), (30, 48, 5), (40, 55, 7)]


def calculate_new_seeds(seed_range, mappings):
    if type(seed_range) == int:
        print(f"Found single value")
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
                print(f"Fully matching")
                mapping_output = [(mapping_dest_start, mapping_dest_end)]
                break
            elif (
                (mapping_src_start < working_seed_start)
                and (mapping_src_end > working_seed_start)
                and (mapping_src_end < working_seed_end)
            ):
                print(f"Overlap at beginning")
                delta = working_seed_start - mapping_src_start
                mapping_output = [(mapping_dest_start + delta, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break
            elif (mapping_src_start < working_seed_start) and (
                mapping_src_end == working_seed_start
            ):
                print(f"Found overlap on first value of seed")
                mapping_output = [mapping_dest_end]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]

                break
            elif (mapping_src_start > working_seed_start) and (
                mapping_src_end < working_seed_end
            ):
                print(f"mapping in the middle")
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapping_output = [lower_range, (mapping_dest_start, mapping_dest_end)]
                remaining_seed_range = [(mapping_src_end + 1, working_seed_end)]
                break

            elif mapping_src_end < working_seed_start:
                print(f"Mapping range too low")
            elif mapping_src_start > working_seed_end:
                print(f"Mapping range too high")
            elif (mapping_src_start > working_seed_start) and (
                mapping_src_end == working_seed_end
            ):
                print(f"mapping in middle, ends on end seed")
                lower_range = (working_seed_start, mapping_src_start - 1)
                mapping_output = [lower_range, (mapping_dest_start, mapping_dest_end)]
                break

    temp_output = []
    for seed_start, seed_end in mapping_output:
        if seed_start == seed_end:
            temp_output.append(seed_start)
        else:
            temp_output.append((seed_start, seed_end))

    if (mapping_output is None) and (remaining_seed_range is None):
        return [seed_range], None

    return mapping_output, remaining_seed_range


if __name__ == "__main__":
    seed_range = []

    for seed_start, seed_range_size in seeds:
        seed_range.append((seed_start, seed_start + seed_range_size))

    seed_range = sorted(seed_range, key=lambda tup: tup[0])

    print(seed_range)
