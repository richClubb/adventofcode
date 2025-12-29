#include "seed_calc.h"

#include <stdint.h>
#include <stdbool.h>

inline bool seed_map_map_seed(uint64_t *value, const SEED_MAP *seed_map)
{
    if (
        (*value >= seed_map->source) && 
        (*value < (seed_map->source + seed_map->size))
    )
    {
        *value = *value - seed_map->source + seed_map->target;
        return true;
    }

    return false;
}

inline uint64_t seed_map_layer_map_seed(const uint64_t value, const SEED_MAP_LAYER *seed_map_layer)
{
    uint64_t temp_value = value;
    for (uint64_t index = 0; index < seed_map_layer->num_seed_maps; index++)
    {
        if (seed_map_map_seed(&temp_value, &seed_map_layer->seed_maps[index]))
        {
            return temp_value;
        }
    }

    return value;
}

uint64_t seed_map_layers_map_seed(const uint64_t value, const SEED_MAP_LAYERS *seed_map_layers)
{
    uint64_t temp_value = value;
    for (uint64_t index = 0; index < seed_map_layers->num_seed_map_layers; index++)
    {
        temp_value = seed_map_layer_map_seed(temp_value, &seed_map_layers->seed_map_layers[index]);
    }
    return temp_value;
}

uint64_t seed_range_find_min(const SEED_RANGE *seed_range, const SEED_MAP_LAYERS *seed_map_layers)
{
    uint64_t min_value = UINT64_MAX;
    for (uint64_t seed = seed_range->start; seed < (seed_range->start + seed_range->size); seed++ )
    {
        uint64_t temp_seed = seed;
        temp_seed = seed_map_layers_map_seed(temp_seed, seed_map_layers);
        if (min_value > temp_seed) min_value = temp_seed;
    }

    return min_value;
}
