#ifndef __SEED_RANGE_H__

#define __SEED_RANGE_H__

#include <stdint.h>

typedef struct seed_range_t {
    unsigned long start;
    unsigned long size;
} SEED_RANGE;

SEED_RANGE **get_seed_ranges(
    const char *line, 
    unsigned int *num_seed_ranges
);

void seed_ranges_term(
    SEED_RANGE **seed_ranges, 
    unsigned int num_seed_ranges
);

SEED_RANGE **seed_ranges_split_by_size(
    SEED_RANGE **seed_ranges, 
    uint32_t num_seed_ranges, 
    uint32_t max_size, 
    uint64_t *num_new_seed_ranges
);

SEED_RANGE **seed_ranges_split_by_count(
    SEED_RANGE **seed_ranges, 
    uint32_t num_seed_ranges, 
    uint32_t max_size, 
    uint64_t *count
);

#endif