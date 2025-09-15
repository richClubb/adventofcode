#ifndef __SEED_RANGE_H__

#define __SEED_RANGE_H__

typedef struct seed_range_t {
    unsigned long start;
    unsigned long size;
} SEED_RANGE;

SEED_RANGE **get_seed_ranges(const char *line, unsigned int *num_seed_ranges);
void seed_ranges_term(SEED_RANGE **seed_ranges, unsigned int num_seed_ranges);

#endif