#include "seed_range.cuh"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "utils.cuh"

SEED_RANGE **get_seed_ranges(const char *line, unsigned int *num_seed_ranges)
{
    // seed line always starts with 'seeds: ' so atrip that out
    unsigned int seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    unsigned int num_numbers = 0;
    unsigned long *numbers = extract_number_list(seeds_substring, &num_numbers);

    SEED_RANGE **seed_ranges = (SEED_RANGE **)calloc(num_numbers / 2, sizeof(SEED_RANGE *));

    for(unsigned int index = 0; index < num_numbers; index += 2)
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

void seed_ranges_term(SEED_RANGE **seed_ranges, unsigned int num_seed_ranges)
{
    for(unsigned int index = 0; index < num_seed_ranges; index++)
    {
        free(seed_ranges[index]);
    }
    free(seed_ranges);
}