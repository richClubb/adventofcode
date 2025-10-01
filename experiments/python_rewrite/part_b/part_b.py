
from seed_map.seed_map import SeedMap
from seed_map_layer.seed_map_layer import SeedMapLayer
from seed_range.seed_range import SeedRange

from multiprocessing import Pool

def extract_seed_ranges_from_string(input_string: str):
    numbers_string = input_string.strip("seeds: ")

    numbers = list(map(int, numbers_string.split()))

    seed_ranges = []
    for index in range(0, len(numbers), 2):
        seed_ranges.append(SeedRange(numbers[index], numbers[index+1]))

    return seed_ranges 

def extract_seed_ranges_and_seed_map_layers(file_path):
    seed_ranges = None
    seed_map_layers = None
    current_seed_map_layer = None
    with open(file_path) as file:

        for line in file:
            if(len(line) <= 2):
                continue
            
            if(line.find("seeds: ") != -1):
                seed_ranges = extract_seed_ranges_from_string(line)
                continue

            if(line.find(":") != -1):
                if seed_map_layers == None and current_seed_map_layer == None:
                    current_seed_map_layer = SeedMapLayer()
                    seed_map_layers = []
                else:
                    seed_map_layers.append(current_seed_map_layer)
                    current_seed_map_layer = None
                    current_seed_map_layer = SeedMapLayer()
                continue

            current_seed_map_layer.add_map(SeedMap.init_from_string(line))
    
        seed_map_layers.append(current_seed_map_layer)

    return seed_ranges, seed_map_layers


def part_b_forward(file_path):
        
    seed_ranges, seed_map_layers = extract_seed_ranges_and_seed_map_layers(file_path)

    min_value = 10**30
    for seed_range in seed_ranges:
        if(result:= seed_range.find_min_in_range(seed_map_layers)) < min_value:
            min_value = result

    return min_value

def part_b_inverse(file_path):
    
    seed_ranges, seed_map_layers = extract_seed_ranges_and_seed_map_layers(file_path)

    start_value = 0
    while(True):
        value = start_value

        for seed_map_layer in reversed(seed_map_layers):
            value = seed_map_layer.map_seed_inverse(value)

        for seed_range in seed_ranges:
            if seed_range.value_in_range(value):
                return start_value

        start_value = start_value + 1


def find_lowest_location(arguments):

    seed_range = arguments[0]
    seed_map_layers = arguments[1]

    min = seed_range.find_min_in_range(seed_map_layers)

    return min

def part_b_forward_parallel(file_path):
    
    seed_ranges, seed_map_layers = extract_seed_ranges_and_seed_map_layers(file_path)

    pool_arguments = []

    for seed_range in seed_ranges:
        pool_arguments.append((seed_range, seed_map_layers))

    with Pool(4) as p:
        results = p.map(find_lowest_location, pool_arguments)

    return min(results)

def part_b_inverse_parallel():
    pass