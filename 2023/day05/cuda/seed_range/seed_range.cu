#include "seed_range.cuh"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "utils.cuh"

SEED_RANGE **get_seed_ranges(const char *line, uint32_t *num_seed_ranges)
{
    // seed line always starts with 'seeds: ' so atrip that out
    uint32_t seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    uint32_t num_numbers = 0;
    uint64_t *numbers = extract_number_list(seeds_substring, &num_numbers);

    SEED_RANGE **seed_ranges = (SEED_RANGE **)calloc(num_numbers / 2, sizeof(SEED_RANGE *));

    for(uint32_t index = 0; index < num_numbers; index += 2)
    {
        SEED_RANGE *seed_range = (SEED_RANGE *)calloc(1, sizeof(SEED_RANGE));

        seed_range->start = numbers[index];
        seed_range->size  = numbers[index + 1];
        seed_ranges[index / 2] = seed_range;
    }

    free(numbers);
    free(seeds_substring);

    *num_seed_ranges = num_numbers / 2;
    return seed_ranges;
}

void seed_ranges_term(SEED_RANGE **seed_ranges, uint32_t num_seed_ranges)
{
    for(uint32_t index = 0; index < num_seed_ranges; index++)
    {
        free(seed_ranges[index]);
    }
    free(seed_ranges);
}

SEED_RANGE **seed_ranges_split_by_size(SEED_RANGE **seed_ranges, uint32_t num_seed_ranges, uint32_t max_size, uint64_t *num_new_seed_ranges)
{
    SEED_RANGE **new_seed_ranges = (SEED_RANGE **)calloc(0, sizeof(SEED_RANGE *));

    uint64_t total_count = 0;

    for (uint32_t index = 0; index < num_seed_ranges; index++)
    {
        SEED_RANGE *curr_seed_range = seed_ranges[index];
        total_count += curr_seed_range->size;
    }

    printf("Total: %lu\n", total_count);

    return NULL;
}

SEED_RANGE **seed_ranges_split_by_count(SEED_RANGE **seed_ranges, uint32_t num_seed_ranges, uint32_t max_size, uint64_t *count)
{
    SEED_RANGE **new_seed_ranges = (SEED_RANGE **)calloc(0, sizeof(SEED_RANGE *));

    uint64_t total_count = 0;

    for (uint32_t index = 0; index < num_seed_ranges; index++)
    {
        SEED_RANGE *curr_seed_range = seed_ranges[index];
        total_count += curr_seed_range->size;
    }

    printf("Total: %lu\n", total_count);

    return NULL;
}