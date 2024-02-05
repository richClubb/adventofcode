import argparse
import os

from copy import copy

RUNS = ["part_a", "part_b_forward", "part_b_backward", "part_b_ranges"]


class Seed:
    def __init__(self, value: int):
        self.__value = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value: int):
        self.__value = value

    def __eq__(self, value):
        if type(value) == int:
            return self.__value == value
        elif type(value) == Seed:
            return self.__value == value.value

    def __ne__(self, value):
        if type(value) == int:
            return self.__value != value
        elif type(value) == Seed:
            return self.__value != value.value

    def __lt__(self, value):
        if type(value) == int:
            return self.__value < value
        elif type(value) == Seed:
            return self.__value < value.value

    def __gt__(self, value):
        if type(value) == int:
            return self.__value > value
        elif type(value) == Seed:
            return self.__value > value.value

    def __le__(self, value):
        if type(value) == int:
            return self.__value <= value
        elif type(value) == Seed:
            return self.__value <= value.value

    def __ge__(self, value):
        if type(value) == int:
            return self.__value >= value
        elif type(value) == Seed:
            return self.__value >= value.value


class SeedRange:
    def __init__(self, start_value: int, size=None, end=None):
        self.__start_value = start_value
        if size is None:
            self.__end_value = end
        elif end is None:
            self.__end_value = start_value + size - 1
        else:
            raise Exception("Must have a size or and end value")
        self.__next_value = self.__start_value

    @property
    def value(self):
        return (self.__start_value, self.__end_value)

    @property
    def start_value(self):
        return self.__start_value

    def in_seed_range(self, value: int):
        if (value >= self.__start_value) and (value <= self.__end_value):
            return True
        return False

    def __eq__(self, comp):
        if type(comp) == tuple:
            return (self.__start_value == comp[0]) and (self.__end_value == comp[1])
        elif type(comp) == SeedRange:
            comp_value = comp.value
            return (self.__start_value == comp_value[0]) and (
                self.__end_value == comp_value[1]
            )

    def __lt__(self, other):
        return self.__start_value < other.start_value

    def __iter__(self):
        self.__next_value = self.__start_value
        return self

    def __next__(self):
        if self.__next_value > self.__end_value:
            raise StopIteration()

        value = self.__next_value
        self.__next_value += 1

        return Seed(value)

    def __str__(self):
        return f"({self.__start_value}, {self.__end_value})"

    def __repr__(self):
        return str(self)


class Mapping:
    def __init__(self, dest: int, src: int, size: int):
        self.__dest = dest
        self.__dest_end = dest + size - 1
        self.__src = src
        self.__src_end = src + size - 1
        self.__size = size

    @property
    def src(self):
        return self.__src

    def map_seed(self, seed: Seed):
        if (seed.value >= self.__src) and (seed.value <= self.__src + self.__size - 1):
            return Seed(self.__dest + (seed.value - self.__src))
        return None

    def map_seed_inverse(self, seed: Seed):
        if (seed.value >= self.__dest) and (
            seed.value <= self.__dest + self.__size - 1
        ):
            return Seed(self.__src + (seed.value - self.__dest))
        return None

    def map_seed_range(self, seed_range: SeedRange):
        """
        calculates the new seed ranges.
        """
        return self.__case_1(seed_range)

    def __lt__(self, other: SeedRange):
        return self.__src < other.src

    def __case_1(self, seed_range: SeedRange):
        _, seed_range_end = seed_range.value
        if seed_range_end < self.__src:
            # print(f"Seed Range {seed_range} Matched case 1")
            return [seed_range], None
        return self.__case_2(seed_range)

    def __case_2(self, seed_range: SeedRange):
        seed_range_start, _ = seed_range.value
        if seed_range_start > self.__src_end:
            # print(f"Seed Range {seed_range} Matched case 2")
            return [], seed_range
        return self.__case_3(seed_range)

    def __case_3(self, seed_range: SeedRange):
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start < self.__src) and (seed_range_end == self.__src):
            # print(f"Seed Range {seed_range} Matched case 3")
            lower_range = SeedRange(seed_range_start, end=self.__src - 1)
            mapped_range = SeedRange(self.__dest, end=self.__dest)
            return [lower_range, mapped_range], None
        return self.__case_4(seed_range)

    def __case_4(self, seed_range: SeedRange):
        """
        Seed overlapping with the beginning of the mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (
            (seed_range_start < self.__src)
            and (seed_range_end > self.__src)
            and (seed_range_end < self.__src_end)
        ):
            # print(f"Seed Range {seed_range} Matched case 4")
            lower_range = SeedRange(seed_range_start, end=self.__src - 1)
            offset = seed_range_end - self.__src
            mapped_range = SeedRange(self.__dest, end=self.__dest + offset)
            return [lower_range, mapped_range], None
        return self.__case_5(seed_range)

    def __case_5(self, seed_range: SeedRange):
        """
        Seed overflowing the beginning of the mapping range but fully encompassed in
        the mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start < self.__src) and (seed_range_end == self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 5")
            lower_range = SeedRange(seed_range_start, end=self.__src - 1)
            mapped_range = SeedRange(self.__dest, end=self.__dest_end)
            return [lower_range, mapped_range], None

        return self.__case_6(seed_range)

    def __case_6(self, seed_range: SeedRange):
        """
        Seed overflowing the start and end of the mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start < self.__src) and (seed_range_end > self.__src):
            # print(f"Seed Range {seed_range} Matched case 6")
            lower_range = SeedRange(seed_range_start, end=self.__src - 1)
            mapped_range = SeedRange(self.__dest, end=self.__dest_end)
            remaining_seed_range = SeedRange(self.__src_end + 1, end=seed_range_end)
            return [lower_range, mapped_range], remaining_seed_range
        return self.__case_7(seed_range)

    def __case_7(self, seed_range: SeedRange):
        """
        Seed starting at the beginning of the mapping range but contained inside
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start == self.__src) and (seed_range_end < self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 7")
            offset = seed_range_end - self.__src
            mapped_range = SeedRange(self.__dest, end=self.__dest + offset)
            return [mapped_range], None
        return self.__case_8(seed_range)

    def __case_8(self, seed_range: SeedRange):
        """
        Seed matches the entire mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start == self.__src) and (seed_range_end == self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 8")
            mapped_range = SeedRange(self.__dest, end=self.__dest_end)
            return [mapped_range], None
        return self.__case_9(seed_range)

    def __case_9(self, seed_range: SeedRange):
        """
        Seed matching the beginning of the mapping range but overflowing the end
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start == self.__src) and (seed_range_end > self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 9")
            mapped_range = SeedRange(self.__dest, end=self.__dest_end)
            return [mapped_range], SeedRange(self.__src_end + 1, end=seed_range_end)
        return self.__case_10(seed_range)

    def __case_10(self, seed_range: SeedRange):
        """
        Seed starts and ends in the middle of the mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start > self.__src) and (seed_range_end < self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 10")
            offset = seed_range_start - self.__src
            size = seed_range_end - seed_range_start
            mapped_range = SeedRange(
                self.__dest + offset,
                end=self.__dest + offset + size,
            )
            return [mapped_range], None
        return self.__case_11(seed_range)

    def __case_11(self, seed_range: SeedRange):
        """
        Seed starts in the middle of the mapping range and ends at the end.
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start > self.__src) and (seed_range_end == self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 11")
            offset = seed_range_start - self.__src
            mapped_range = SeedRange(self.__dest + offset, end=self.__dest_end)
            return [mapped_range], None
        return self.__case_12(seed_range)

    def __case_12(self, seed_range: SeedRange):
        """
        Seed starts in the middle of the mapping range but overflows the end
        """
        seed_range_start, seed_range_end = seed_range.value
        if (
            (seed_range_start > self.__src)
            and (seed_range_start < self.__src_end)
            and (seed_range_end > self.__src_end)
        ):
            # print(f"Seed Range {seed_range} Matched case 12")
            offset = seed_range_start - self.__src
            mapped_range = SeedRange(self.__dest + offset, end=self.__dest_end)
            return [mapped_range], SeedRange(self.__src_end + 1, end=seed_range_end)
        return self.__case_13(seed_range)

    def __case_13(self, seed_range: SeedRange):
        """
        Seed overlaps with the last value of the mapping range
        """
        seed_range_start, seed_range_end = seed_range.value
        if (seed_range_start == self.__src_end) and (seed_range_end > self.__src_end):
            # print(f"Seed Range {seed_range} Matched case 13")
            mapped_range = SeedRange(self.__dest_end, end=self.__dest_end)
            return [mapped_range], SeedRange(seed_range_start + 1, end=seed_range_end)
        raise Exception("Could not match a case")


class MappingLayer:
    def __init__(self):
        self.__mappings = []

    def add_mapping(self, mapping: Mapping):
        self.__mappings.append(mapping)

    def map_seed(self, seed: Seed):
        for mapping in self.__mappings:
            if result := mapping.map_seed(seed):
                return result
        return None

    def map_seed_inverse(self, seed: Seed) -> Seed:
        for mapping in self.__mappings:
            if result := mapping.map_seed_inverse(seed):
                return result
        return None

    def __map_seed_range(self, seed_range):

        unmapped_range = seed_range
        mapped_seed_ranges = []
        for mapping in self.__mappings:
            mapped, unmapped_range = mapping.map_seed_range(unmapped_range)

            if len(mapped) > 0:
                mapped_seed_ranges += mapped

            if unmapped_range is None:
                break

        return mapped_seed_ranges, unmapped_range

    def map_seed_ranges(self, seed_ranges):
        # Pre-sort the seed ranges
        unmapped_seed_ranges = copy(seed_ranges)
        unmapped_seed_ranges.sort()
        mapped_seed_ranges = []
        self.__mappings.sort()

        while len(unmapped_seed_ranges) > 0:
            curr_seed = unmapped_seed_ranges.pop(0)
            mapped, unmapped = self.__map_seed_range(curr_seed)

            if unmapped:
                unmapped_seed_ranges.append(unmapped)
                unmapped_seed_ranges.sort()

            if len(mapped) > 0:
                mapped_seed_ranges += mapped
            else:
                mapped_seed_ranges.append(unmapped_seed_ranges.pop(0))

        mapped_seed_ranges.sort()
        return mapped_seed_ranges


class MappingLayers:
    def __init__(self):
        self.__mapping_layers = []

    def add_mapping_layer(self, mapping_layer: MappingLayer):
        self.__mapping_layers.append(mapping_layer)

    def map_seed(self, seed: Seed) -> Seed:
        for mapping_layer in self.__mapping_layers:
            if result := mapping_layer.map_seed(seed):
                seed = result
        return seed

    def map_seed_inverse(self, seed: Seed) -> Seed:
        start = len(self.__mapping_layers) - 1
        reversed_mapping_list = [self.__mapping_layers[i] for i in range(start, -1, -1)]
        value = seed
        for mapping_layer in reversed_mapping_list:
            if result := mapping_layer.map_seed_inverse(value):
                value = result
        return value

    def map_seed_ranges(self, seed_ranges: SeedRange):

        index = 0
        for mapping_layer in self.__mapping_layers:
            index += 1
            seed_ranges = mapping_layer.map_seed_ranges(seed_ranges)

        return seed_ranges


def get_raw_seeds_from_file(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seeds"):
                return list(map(int, line.split()[1:]))


def load_seeds_from_file(path):
    seeds_raw = get_raw_seeds_from_file(path)
    return [Seed(seed) for seed in seeds_raw]


def load_seed_ranges_from_file(path):
    seeds_raw = get_raw_seeds_from_file(path)
    f = lambda A, n=3: [A[i : i + n] for i in range(0, len(A), n)]
    seed_pairs = f(seeds_raw, 2)
    seed_ranges = [SeedRange(seed_pair[0], seed_pair[1]) for seed_pair in seed_pairs]

    return seed_ranges


def load_mapping_layers_from_file(path):
    mapping_layers = MappingLayers()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seeds"):
                continue

            elif line.startswith("seed-to-soil"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("soil-to-fertilizer"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("fertilizer-to-water"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("water-to-light"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("light-to-temperature"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("temperature-to-humidity"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line.startswith("humidity-to-location"):
                mapping_layer = MappingLayer()
                mapping_layers.add_mapping_layer(mapping_layer)
            elif line:
                dest, src, size = list(map(int, line.split()))
                mapping_layer.add_mapping(Mapping(dest, src, size))

    return mapping_layers


def part_a(path):
    seeds = load_seeds_from_file(path)
    mapping_layers = load_mapping_layers_from_file(path)

    min_value = 10**30
    for seed in seeds:
        result = mapping_layers.map_seed(seed)
        if result < min_value:
            min_value = result.value

    return min_value


def part_b_forwards(path):
    seed_ranges = load_seed_ranges_from_file(path)
    mapping_layers = load_mapping_layers_from_file(path)

    min_value = 10**30
    for seed_range in seed_ranges:
        for seed in seed_range:
            result = mapping_layers.map_seed(seed)
            if result < min_value:
                min_value = result.value

    return min_value


def part_b_backward(path):
    seed_ranges = load_seed_ranges_from_file(path)
    mapping_layers = load_mapping_layers_from_file(path)

    value = 0
    found_result = False
    while found_result == False:
        seed = Seed(value)
        result = mapping_layers.map_seed_inverse(seed)

        for seed_range in seed_ranges:
            if seed_range.in_seed_range(result):
                return value

        value += 1


def part_b_ranges(path):
    seed_ranges = load_seed_ranges_from_file(path)
    mapping_layers = load_mapping_layers_from_file(path)

    seed_ranges.sort()
    result = mapping_layers.map_seed_ranges(seed_ranges)

    min_value = result[0].value[0]
    return min_value


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
        print(f"part b forward: {part_b_forwards(args.input_file_path)}")
    elif args.run == "part_b_backward":
        print(f"part b backwards: {part_b_backward(args.input_file_path)}")
    elif args.run == "part_b_ranges":
        print(f"part b ranges: {part_b_ranges(args.input_file_path)}")
