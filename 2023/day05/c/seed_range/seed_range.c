#include "seed_range.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "utils.h"

#define SEEDS_LINE_PREFIX "seeds: "
#define SEEDS_LINE_PREFIX_LEN (sizeof(SEEDS_LINE_PREFIX) - 1)

SEED_RANGE *get_seed_ranges(const char *line, uint64_t *num_seed_ranges)
{
    if (strstr(line, SEEDS_LINE_PREFIX) == NULL)
    {
        return NULL;
    }

    // sizeof(SEEDS_LINE_PREFIX) needs to subtract 1 as it has a string terminator
    unsigned int seeds_substring_len = strlen(line) - SEEDS_LINE_PREFIX_LEN;
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    assert(seeds_substring != NULL);

    strncpy(seeds_substring, line + SEEDS_LINE_PREFIX_LEN, seeds_substring_len);

    uint64_t num_numbers = 0;
    uint64_t *numbers = extract_number_list(seeds_substring, &num_numbers);

    free(seeds_substring);

    *num_seed_ranges = num_numbers / 2;

    SEED_RANGE *seed_ranges = (SEED_RANGE *)calloc((*num_seed_ranges + 1), sizeof(SEED_RANGE));

    for(uint64_t index = 0; index < num_numbers; index += 2)
    {
        SEED_RANGE temp = {
            .start = numbers[index],
            .size = numbers[index + 1]
        };

        seed_ranges[index/2] = temp;
    }

    seed_ranges[*num_seed_ranges].size = 0;
    seed_ranges[*num_seed_ranges].start = 0;

    free(numbers);

    return seed_ranges;
}

void seed_ranges_term(SEED_RANGE **seed_ranges, uint64_t num_seed_ranges)
{
    for(unsigned int index = 0; index < num_seed_ranges; index++)
    {
        free(seed_ranges[index]);
    }
    free(seed_ranges);
}

void sort_seed_ranges_by_size(SEED_RANGE *seed_ranges, uint64_t num_seed_ranges)
{
    for(uint64_t index_1 = 0; index_1 < num_seed_ranges; index_1++)
    {
        for(uint64_t index_2 = 0; index_2 < num_seed_ranges - index_1 - 1; index_2++)
        {
            SEED_RANGE range_a = seed_ranges[index_2];
            SEED_RANGE range_b = seed_ranges[index_2 + 1];

            if( range_a.size > range_b.size)
            {
                SEED_RANGE temp = range_a;
                seed_ranges[index_2] = range_b;
                seed_ranges[index_2 + 1] = temp;
            }
        }
    }
}

void sort_seed_ranges_by_start(SEED_RANGE *seed_ranges, uint64_t num_seed_ranges)
{
    for(uint64_t index_1 = 0; index_1 < num_seed_ranges; index_1++)
    {
        for(uint64_t index_2 = 0; index_2 < num_seed_ranges - index_1 - 1; index_2++)
        {
            SEED_RANGE range_a = seed_ranges[index_2];
            SEED_RANGE range_b = seed_ranges[index_2 + 1];

            if( range_a.start > range_b.start)
            {
                SEED_RANGE temp = range_a;
                seed_ranges[index_2] = range_b;
                seed_ranges[index_2 + 1] = temp;
            }
        }
    }
}

SEED_RANGE *split_seed_range_by_size(SEED_RANGE *seed_range, uint64_t max_size, uint64_t *range_count)
{
    if (seed_range->size <= max_size)
    {
        SEED_RANGE *new_seed_range = (SEED_RANGE *)calloc(1, sizeof(SEED_RANGE));
        *new_seed_range = *seed_range;
        *range_count = 1;
        return new_seed_range;
    }

    uint64_t seed_range_count = 0;
    SEED_RANGE *new_seed_ranges = (SEED_RANGE *)calloc(1, sizeof(SEED_RANGE));
    uint64_t remaining = seed_range->size;
    uint64_t curr_start = seed_range->start;
    while(remaining > 0)
    {
        uint64_t curr_ideal_size = ideal_size(remaining, max_size);
        if(curr_ideal_size <= remaining)
        {
            new_seed_ranges = (SEED_RANGE *)realloc(new_seed_ranges, (seed_range_count +  2) * sizeof(SEED_RANGE));
            new_seed_ranges[seed_range_count].start = curr_start;
            new_seed_ranges[seed_range_count].size = curr_ideal_size;
            remaining -= curr_ideal_size;
            curr_start += curr_ideal_size;
        }
        else
        {
            new_seed_ranges = (SEED_RANGE *)realloc(new_seed_ranges, (seed_range_count +  2) * sizeof(SEED_RANGE));
            new_seed_ranges[seed_range_count].start = curr_start;
            new_seed_ranges[seed_range_count].size = remaining;
            remaining = 0;
        }
        seed_range_count += 1;
    }

    * range_count = seed_range_count;
    return new_seed_ranges;
}

SEED_RANGE *split_seed_ranges_by_number(SEED_RANGE *seed_ranges, uint64_t *num_seed_ranges, uint64_t max_seed_ranges)
{
    uint64_t total_count = 0;

    SEED_RANGE *new_seed_range = (SEED_RANGE *)calloc(0, sizeof(SEED_RANGE));
    
    for(uint64_t index = 0; index < *num_seed_ranges; index++)
    {
        total_count += seed_ranges[index].size;
    }

    uint64_t remaining = total_count;
    uint64_t seed_range_count = 0;
    for(uint64_t index = 0; index < *num_seed_ranges; index++)
    {
        uint64_t curr_ideal_size = remaining / (max_seed_ranges - seed_range_count);
        uint64_t curr_ideal_size_rem = remaining % (max_seed_ranges - seed_range_count);

        if (curr_ideal_size_rem != 0) curr_ideal_size += 1;

        uint64_t temp_range_count = 0;
        SEED_RANGE *temp_seed_range = split_seed_range_by_size(&seed_ranges[index], curr_ideal_size, &temp_range_count);        
        
        new_seed_range = (SEED_RANGE *)realloc(new_seed_range, (seed_range_count + temp_range_count) * sizeof(SEED_RANGE));
        memcpy((new_seed_range + (seed_range_count)), temp_seed_range, temp_range_count * sizeof(SEED_RANGE));
        seed_range_count += temp_range_count;

        remaining -= seed_ranges[index].size;
        free(temp_seed_range);
    }

    *num_seed_ranges = seed_range_count;
    return new_seed_range;
}

uint64_t ideal_size(uint64_t size, uint64_t max_partition_size)
{
    if (max_partition_size == 1)
    {
        return 1;
    }

    if (size <= max_partition_size)
    {
        return size;
    }

    uint64_t initial_attempt = size / max_partition_size;
    uint64_t initial_attempt_rem = size % max_partition_size;

    if (initial_attempt_rem == 0)
    {
        return max_partition_size;
    }

    if ((initial_attempt == 2) && (initial_attempt_rem == 0))
    {
        return size / 2;
    }

    uint64_t divisor = initial_attempt + 1;
    uint64_t new_size_initial = size / divisor;
    uint64_t new_size_initial_rem = size % divisor;

    if (new_size_initial_rem == 0)
    {
        return new_size_initial;
    }
    
    return new_size_initial + 1;
}