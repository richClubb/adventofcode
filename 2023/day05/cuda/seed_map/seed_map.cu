#include "seed_map.cuh"

#include "utils.cuh"

SEED_MAP *get_seed_map(char *line)
{
    SEED_MAP *seed_map = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));

    unsigned int length = 0;
    unsigned long *map_values = extract_number_list(line, &length);

    seed_map->source = map_values[1];
    seed_map->target = map_values[0];
    seed_map->size   = map_values[2];

    free(map_values);

    return seed_map;
}