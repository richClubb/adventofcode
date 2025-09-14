#include "seed_map.h"

#include <stdbool.h>

bool seed_map_map_seed(const SEED_MAP *seed_map, long *seed_value)
{
    if (
        (*seed_value >= seed_map->source) && 
        (*seed_value < (seed_map->source + seed_map->size))
    )
    {
        *seed_value = (*seed_value - seed_map->source) + seed_map->target;
        return true;
    }

    return false;
}

