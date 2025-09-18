#ifndef __SEED_MAP_CUH__

#define __SEED_MAP_CUH__

#include <stdint.h>

typedef struct seed_map_t
{
    uint64_t source;
    uint64_t target;
    uint64_t size;
} SEED_MAP;

#endif

