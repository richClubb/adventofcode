#include "seed_map.h"

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>

#include "utils.h"

void seed_map_init(SEED_MAP **seed_map)
{
    assert(*seed_map == NULL);

    *seed_map = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
}

void seed_map_term(SEED_MAP *seed_map)
{
    free(seed_map);
    seed_map = NULL;
}

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

bool seed_map_map_seed(const SEED_MAP *seed_map, unsigned long *seed_value)
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

