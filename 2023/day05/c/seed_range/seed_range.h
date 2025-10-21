#ifndef __SEED_RANGE_H__

#define __SEED_RANGE_H__

#include <stdint.h>

typedef struct seed_range_t {
    uint64_t start;
    uint64_t size;
} SEED_RANGE;

SEED_RANGE *get_seed_ranges(const char *line, uint64_t *num_seed_ranges);
void seed_ranges_term(SEED_RANGE **seed_ranges, uint64_t num_seed_ranges);

void sort_seed_ranges_by_size(SEED_RANGE *seed_ranges, uint64_t num_seed_ranges);

void sort_seed_ranges_by_start(SEED_RANGE *seed_ranges, uint64_t num_seed_ranges);

SEED_RANGE *split_seed_range_by_size(SEED_RANGE *seed_range, uint64_t max_size, uint64_t *range_count);

SEED_RANGE *split_seed_ranges_by_number(SEED_RANGE *seed_ranges, uint64_t *num_seed_ranges, uint64_t max_seed_ranges);

uint64_t ideal_size(uint64_t size, uint64_t max_partition_size);

#endif