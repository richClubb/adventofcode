#include "seed.cuh"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <stdio.h>

#define SEEDS_LINE_PREFIX "seeds: "
#define SEEDS_LINE_PREFIX_LEN (sizeof(SEEDS_LINE_PREFIX) - 1)

uint64_t *extract_number_list(const char *number_string, uint64_t *length);

SEED *seed_get_seeds_from_line(const char* line, uint64_t *num_seeds)
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

    uint64_t *seeds = extract_number_list(seeds_substring, num_seeds);

    free(seeds_substring);

    for(uint64_t index_1 = 0; index_1 < *num_seeds; index_1++)
    {
        for(uint64_t index_2 = 0; index_2 < *num_seeds - index_1 - 1; index_2++)
        {
            uint64_t seed_1 = seeds[index_2];
            uint64_t seed_2 = seeds[index_2 + 1];

            if(seed_1 > seed_2)
            {
                uint64_t seed_temp = seed_1;
                seeds[index_2] = seed_2;
                seeds[index_2 + 1] = seed_temp;
            }
        }
    }

    return seeds;
}