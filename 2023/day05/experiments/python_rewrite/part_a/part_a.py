from seed_map.seed_map import SeedMap
from seed_map_layer.seed_map_layer import SeedMapLayer


def extract_seeds_from_string(input_string: str):
    numbers_string = input_string.strip("seeds: ")

    return list(map(int, numbers_string.split()))


def part_a(file_path: str):
    
    seeds = None
    seed_map_layers = None
    current_seed_map_layer = None
    with open(file_path) as file:

        for line in file:
            if(len(line) <= 2):
                continue
            
            if(line.find("seeds: ") != -1):
                seeds = extract_seeds_from_string(line)
                continue

            if(line.find(":") != -1):
                if seed_map_layers == None and current_seed_map_layer == None:
                    current_seed_map_layer = SeedMapLayer()
                    seed_map_layers = []
                else:
                    current_seed_map_layer.sort_maps()
                    seed_map_layers.append(current_seed_map_layer)
                    current_seed_map_layer = None
                    current_seed_map_layer = SeedMapLayer()
                continue

            current_seed_map_layer.add_map(SeedMap.init_from_string(line))
        
        current_seed_map_layer.sort_maps()
        seed_map_layers.append(current_seed_map_layer)

    min_value = 10**30
    for seed in seeds:
        value = seed
        for seed_map_layer in seed_map_layers:
            value = seed_map_layer.map_seed(value)
        if value < min_value:
            min_value = value

    return min_value
