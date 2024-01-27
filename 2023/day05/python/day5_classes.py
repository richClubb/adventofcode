import argparse
import os

from collections import defaultdict
from enum import Enum

RUNS = ["part_a", "part_b_forward", "part_b_backward"]


class SeedType(Enum):
    SEED_SINGLE = 0
    SEED_RANGE = 1


class Seed:
    def __init__(self, seed_start, seed_size=None):
        self.__seed_start = seed_start
        self.__seed_end = seed_size if seed_size is None else seed_start + seed_size - 1
        self.__seed_type = (
            SeedType.SEED_SINGLE if seed_size is None else SeedType.SEED_RANGE
        )

    def seed_type(self):
        return self.__seed_type

    def value(self):
        if self.__seed_type == SeedType.SEED_SINGLE:
            return self.__seed_start
        elif self.__seed_type == SeedType.SEED_RANGE:
            return (self.__seed_start, self.__seed_end)
        else:
            raise Exception("Cannot return value for seed type")

    def set_value(self, seed_start, seed_end=None):
        self.__seed_start = seed_start
        self.__seed_end = seed_end

    def __eq__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start == seed.value:
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

    def __ne__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start != seed.value:
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

    def __lt__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start < seed.value():
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

    def __gt__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start > seed.value:
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

    def __le__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start <= seed.value:
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

    def __ge__(self, seed):
        if self.__seed_type == SeedType.SEED_SINGLE:
            if self.__seed_start >= seed.value:
                return True
            return False
        elif self.__seed_type == SeedType.SEED_RANGE:
            raise Exception("Not yet implemented")

class Seeds:
    def __init__(self, path = None, seed_type = None):
        self.seeds = []
        self.index = 0
        if path is not None:
            self.load_seeds_from_file(path, seed_type)

    def load_seeds_from_file(self, path, seed_type):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("seeds"):
                    seeds_raw = list(map(int, line.split()[1:]))
                    if seed_type == SeedType.SEED_SINGLE:
                        self.seeds = [Seed(seed) for seed in seeds_raw]
                    elif seed_type == SeedType.SEED_RANGE:
                        raise Exception("Not yet implemented")
                    break

    def __iter__(self):
        for each in self.seeds:
              yield each


class SeedMapping:
    def __init__(self, dest, start, size):
        self.__dest = dest
        self.__start = start
        self.__size = size

    def translate_seed(self, seed):
        if seed.seed_type() == SeedType.SEED_SINGLE:
            if (seed.value() >= self.__start) and (
                seed.value() <= (self.__start + self.__size - 1)
            ):
                seed.set_value(self.__dest + (seed.value() - self.__start))

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
    
class MappingLayers:
    def __init__(self, path = None):
        self.mapping_layers = defaultdict(MappingLayer)
        if path is not None:
            self.load_mappings_layers_from_file(path)


    def load_mappings_layers_from_file(self, path):

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("seeds"):
                    continue

                elif line.startswith("seed-to-soil"):
                    active_map = 0
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("soil-to-fertilizer"):
                    active_map = 1
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("fertilizer-to-water"):
                    active_map = 2
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("water-to-light"):
                    active_map = 3
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("light-to-temperature"):
                    active_map = 4
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("temperature-to-humidity"):
                    active_map = 5
                    self.mapping_layers[active_map] = MappingLayer()
                elif line.startswith("humidity-to-location"):
                    active_map = 6
                    self.mapping_layers[active_map] = MappingLayer()
                elif line:
                    dest, start, size = list(map(int, line.split()))
                    self.mapping_layers[active_map].add_seed_map(SeedMapping(dest, start, size))
    
    def translate_seed(self, seed):

        for index in range(0, len(self.mapping_layers.keys())):
            self.mapping_layers[index].translate_seed(seed)


def part_a(path):
    seeds = Seeds(path, SeedType.SEED_SINGLE)

    mapping_layers = MappingLayers(path)

    min_seed = Seed(10**30)

    for seed in seeds:
        mapping_layers.translate_seed(seed)

        if seed < min_seed:
            min_seed = seed

    return seed.value()

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
