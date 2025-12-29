from seed_map_layer import seed_map_layer



class SeedRange:

    def __init__(self, start: int, size: int):
        self.start = start
        self.end = start + size
        self.size = size
        pass

    def value_in_range(self, value):
        if value >= self.start and value < self.end:
            return True
        return False

    def find_min_in_range(self, seed_map_layers):
        
        min_value = 10**30
        for seed in range(self.start, self.end):
            value = seed
            for seed_map_layer in seed_map_layers:
                value = seed_map_layer.map_seed(value)
            if value < min_value:
                min_value = value

        return min_value
    
    def split_into_chunks(self, chunk_size):
        new_seed_ranges = []

        remaining = self.size
        current_start = self.start
        while(remaining > 0):

            if(remaining >= chunk_size):
                new_seed_ranges.append(SeedRange(current_start, chunk_size))
                remaining = remaining - chunk_size
                current_start = current_start + chunk_size
            else:
                new_seed_ranges.append(SeedRange(current_start, remaining))
                remaining = 0

        return new_seed_ranges
