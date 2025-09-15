#ifndef __SEED_MAP_H__

#define __SEED_MAP_H__

#include <stdbool.h>
#include <stdlib.h>

typedef struct seed_map_t
{
    unsigned long source;
    unsigned long target;
    unsigned long size;
} SEED_MAP;

SEED_MAP *get_seed_map(char *line);

void seed_map_init(SEED_MAP **seed_map);

void seed_map_term(SEED_MAP *seed_map);

bool seed_map_map_seed(const SEED_MAP *seed_map, unsigned long *seed_value);

#endif