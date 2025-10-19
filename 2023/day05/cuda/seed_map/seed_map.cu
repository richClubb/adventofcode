#include "seed_map.cuh"

#include <stdint.h>

#include "utils.cuh"

SEED_MAP *seed_map_from_string(char *line)
{
    SEED_MAP *seed_map = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));

    uint64_t length = 0;
    uint64_t *map_values = extract_number_list(line, &length);

    if (length != 3)
    {
        return NULL;
    }

    seed_map->size         = map_values[2];
    seed_map->source_start = map_values[1];
    seed_map->source_end   = seed_map->source_start + seed_map->size;
    seed_map->target_start = map_values[0];
    seed_map->target_end   = seed_map->target_start + seed_map->size;

    free(map_values);

    return seed_map;
}

bool seed_map_map_seed(SEED_MAP *seed_map, SEED *seed)
{
    if(
        (*seed >= seed_map->source_start) && 
        (*seed < seed_map->source_end)
    )
    {
        *seed = *seed - seed_map->source_start + seed_map->target_start;
        return true;
    }

    return false;
}