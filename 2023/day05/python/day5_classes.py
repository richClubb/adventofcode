import argparse

from collections import defaultdict
from enum import Enum

RUNS = ["part_a", "part_b_forward", "part_b_backward"]


class SeedType(Enum):
    SEED_SINGLE = 0
    SEED_RANGE = 1


class Seed:
    def __init__(self, seed_start, seed_size=None):
        self.__seed_start = seed_start
        self.__seed_end = seed_start + self.seed_size - 1
        self.__seed_type = (
            SeedType.SEED_SINGLE if seed_size is None else SeedType.SEED_RANGE
        )

    def seed_type(self):
        return self.__seed_type

    def value(self):
        if self.seed_type == SeedType.SEED_SINGLE:
            return self.__seed_start
        elif self.seed_type == SeedType.SEED_RANGE:
            return (self.__seed_start, self.__seed_end)
        else:
            raise Exception("Cannot return value for seed type")

    def set_value(self, seed_start, seed_end=None):
        self.__seed_start = seed_start
        self.__seed_end = seed_end


class SeedMapping:
    def __init__(self, dest, start, size):
        self.__dest = dest
        self.__start = start
        self.__size = size

    def translate_seed(self, seed):
        if seed.seed_type == SeedType.SEED_SINGLE:
            if (seed.value() >= self.__start) and (
                seed.value() <= (self.__start + self.__size - 1)
            ):
                seed.set_value(self.__dest + (seed.value() - self.start))

    def translate_seed_range(self, seed_range):
        raise Exception("Not yet implemented")


class MappingLayer:
    def __init__(self):
        self.seed_maps = []

    def add_seed_map(self, seed_map):
        self.seed_maps.append(seed_map)

    def translate_seed(self, seed):
        for seed_map in self.seed_maps:
            result = seed_map.translate_seed(seed)
            if result is not None:
                return result
        return seed

    def translate_seed_range(self, seed_range):
        raise Exception("Not yet implemented")


def extract_maps_and_seeds(input_file_path, seed_type):
    maps = defaultdict(list)

    with open(input_file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("seeds"):
                seeds_raw = list(map(int, line.split()[1:]))
                if seed_type == SeedType.SEED_SINGLE:
                    seeds = [Seed(seed) for seed in seeds_raw]
                elif seed_type == SeedType.SEED_RANGE:
                    raise Exception("Not yet implemented")

            elif line.startswith("seed-to-soil"):
                active_map = 0
                maps[active_map] = MappingLayer()
            elif line.startswith("soil-to-fertilizer"):
                active_map = 1
                maps[active_map] = MappingLayer()
            elif line.startswith("fertilizer-to-water"):
                active_map = 2
                maps[active_map] = MappingLayer()
            elif line.startswith("water-to-light"):
                active_map = 3
                maps[active_map] = MappingLayer()
            elif line.startswith("light-to-temperature"):
                active_map = 4
                maps[active_map] = MappingLayer()
            elif line.startswith("temperature-to-humidity"):
                active_map = 5
                maps[active_map] = MappingLayer()
            elif line.startswith("humidity-to-location"):
                active_map = 6
                maps[active_map] = MappingLayer()
            elif line:
                maps[active_map].add_seed_map(list(map(int, line.split())))

    return maps, seeds


def part_a(path):
    pass


def part_b_forwards(path):
    pass


def part_b_backward(path):
    pass


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
