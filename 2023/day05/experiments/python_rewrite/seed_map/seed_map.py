


class SeedMap:

    def __init__(self, source, target, size):
        self.source = source
        self.source_end = source + size
        self.target = target
        self.target_end = target + size
        self.size = size

    def init_from_string(input_string: str):
        number_strings = input_string.split(" ")
        return SeedMap(int(number_strings[1]), int(number_strings[0]), int(number_strings[2]))

    def map_seed(self, value:int):
        if(value >= self.source) and (value < self.source_end):
            return (value - self.source) + self.target
        return None

    def map_seed_inverse(self, value:int):
        if(value >= self.target) and (value < self.target_end):
            return value - self.target + self.source