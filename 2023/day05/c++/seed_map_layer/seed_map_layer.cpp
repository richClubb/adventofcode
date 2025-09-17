#include "seed_map_layer.h"

#include <stdint.h>
#include <optional>

#include "seed_map.h"

SeedMapLayer::SeedMapLayer()
{

}

SeedMapLayer::SeedMapLayer(std::string)
{

}

SeedMapLayer::~SeedMapLayer()
{

}

void SeedMapLayer::add_seed_map(SeedMap seed_map)
{
    this->seed_maps.push_back(seed_map);
}

std::optional<uint64_t> SeedMapLayer::map_seed(uint64_t input)
{
    if (this->seed_maps.size() == 0) return std::nullopt;

    // old-school method
    // for(uint64_t index = 0; index < this->seed_maps.size(); index++)
    // {    
    //     if (std::optional<uint64_t> result; result = this->seed_maps[index]->map_seed(input))
    //     {
    //         return result;
    //     }
    // }

    // iterator method
    // std::vector<SeedMap *>::iterator seedMapIter;
    // for (
    //     seedMapIter = this->seed_maps.begin();
    //     seedMapIter != this->seed_maps.end();
    //     seedMapIter++
    // )
    // {
    //     SeedMap *curr_map = *seedMapIter;
    //     if (std::optional<uint64_t> result; result = curr_map->map_seed(input))
    //     {
    //         return result;
    //     }
    // }

    // more modern method
    for(SeedMap &seed_map : this->seed_maps)
    {
        if (std::optional<uint64_t> result; result = seed_map.map_seed(input))
        {
            return result;
        }
    }

    return std::nullopt;
}

// need a sorting algorithm
void SeedMapLayer::sort_seed_maps()
{
    bool swapped = false;
  
    uint64_t seed_maps_size = this->seed_maps.size();
    
    for (int index_i = 0; index_i < seed_maps_size - 1; index_i++) {
        swapped = false;
        for (int index_j = 0; index_j < seed_maps_size - index_i - 1; index_j++) {
            if (this->seed_maps[index_j].get_source() > this->seed_maps[index_j + 1].get_source()) {
                std::swap(this->seed_maps[index_j], this->seed_maps[index_j + 1]);
                swapped = true;
            }
        }
      
        // If no two elements were swapped, then break
        if (!swapped)
            break;
    }
}
