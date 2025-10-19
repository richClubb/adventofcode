#ifndef __SEED_MAP_H__

#define __SEED_MAP_H__

#include "seed.cuh"

typedef struct seed_map_t {
    uint64_t source_start;
    uint64_t source_end;
    uint64_t target_start;
    uint64_t target_end;
    uint64_t size;
} SEED_MAP;

SEED_MAP *seed_map_from_string(char *line);

bool seed_map_map_seed(SEED_MAP *seed_map, SEED *seed);

#endif