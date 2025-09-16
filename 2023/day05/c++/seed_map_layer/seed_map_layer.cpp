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
    // change this if possible
    for (uint32_t index = 0; index < this->seed_maps.size(); index++)
    {
        delete this->seed_maps[index];
    }
}

void SeedMapLayer::add_seed_map(SeedMap *seed_map)
{
    this->seed_maps.push_back(seed_map);
}

std::optional<uint32_t> SeedMapLayer::map_seed(uint32_t input)
{
    if (this->seed_maps.size() == 0) return std::nullopt;

    // would prefer this just because
    // std::vector<SeedMap>::iterator seedMapIter;
    // for (
    //     seedMapIter = this->seed_maps.begin();
    //     seedMapIter != this->seed_maps.end();
    //     seedMapIter++
    // )
    // {
    //     printf("test\n");
    // }

    for(uint32_t index = 0; index < this->seed_maps.size(); index++)
    {
        std::optional<uint32_t> result;
        if (result = this->seed_maps[index]->map_seed(input))
        {
            return result;
        }
    }

    return std::nullopt;
}

// need a sorting algorithm
