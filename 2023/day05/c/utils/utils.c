#include "utils.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// manually interprets the line
uint64_t *get_seeds(const char *line, uint64_t *num_seeds)
{
    // seed line always starts with 'seeds: ' so atrip that out
    uint64_t seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    // allocate the inital seeds array
    uint64_t *seeds = extract_number_list(seeds_substring, num_seeds);
    // free seeds_substring as we no longer need it.
    free(seeds_substring);

    return seeds;
}

uint64_t *extract_number_list(const char *number_string, uint64_t *length)
{
    char *mut_line = (char *)calloc(strlen(number_string), sizeof(char));
    memcpy(mut_line, number_string, sizeof(char) * strlen(number_string));

    uint64_t *seeds = (uint64_t *)calloc(1, sizeof(uint64_t));
    assert(seeds != NULL);

    uint64_t seeds_index = 0;

    char* token = strtok(mut_line, " ");
    assert(token != NULL);

    while (token != NULL) {
        char* endptr = NULL;

        seeds[seeds_index] = (uint64_t)strtol(token, &endptr, 10);
        seeds_index += 1;
        seeds = (uint64_t *)realloc(seeds, sizeof(uint64_t) * (seeds_index + 1));

        token = strtok(NULL, " ");
    }

    free(mut_line);

    *length = seeds_index;
    return seeds;
}