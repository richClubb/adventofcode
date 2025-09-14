#ifndef __SEED_MAP_H__

#define __SEED_MAP_H__

#include <stdbool.h>

typedef struct seed_map_t
{
    unsigned long source;
    unsigned long target;
    unsigned long size;
} SEED_MAP;

bool seed_map_map_seed(const SEED_MAP *seed_map, long *seed_value);

#endif