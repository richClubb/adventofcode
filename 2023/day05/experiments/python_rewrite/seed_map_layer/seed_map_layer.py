
from seed_map.seed_map import SeedMap

class SeedMapLayer:

    def __init__(self):
        # seeds to be a seed_map
        self.seed_maps:SeedMap = []
        pass

    def add_map(self, seed_map):
        self.seed_maps.append(seed_map)
        pass

    def sort_maps(self):
        self.seed_maps = sorted(self.seed_maps, key=lambda x: x.source)

    def map_seed(self, value: int):

        for seed_map in self.seed_maps:
            if (result:= seed_map.map_seed(value)) is not None:
                return result
        
        return value
    
    def map_seed_inverse(self, value):
        for seed_map in self.seed_maps:
            if(result:= seed_map.map_seed_inverse(value)) is not None:
                return result
            
        return value