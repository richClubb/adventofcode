#include "utils.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

// manually interprets the line
unsigned long *get_seeds(const char *line, unsigned int *num_seeds)
{
    // seed line always starts with 'seeds: ' so atrip that out
    unsigned int seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    // allocate the inital seeds array
    unsigned long *seeds = extract_number_list(seeds_substring, num_seeds);
    // free seeds_substring as we no longer need it.
    free(seeds_substring);

    return seeds;
}

unsigned long *extract_number_list(const char *number_string, unsigned int *length)
{
    char *mut_line = (char *)calloc(strlen(number_string), sizeof(char));
    memcpy(mut_line, number_string, sizeof(char) * strlen(number_string));

    unsigned long *seeds = (unsigned long *)calloc(1, sizeof(unsigned long));
    assert(seeds != NULL);

    unsigned int seeds_index = 0;

    char* token = strtok(mut_line, " ");
    assert(token != NULL);

    while (token != NULL) {
        char* endptr = NULL;

        seeds[seeds_index] = (unsigned long)strtol(token, &endptr, 10);
        seeds_index += 1;
        seeds = (unsigned long *)realloc(seeds, sizeof(unsigned long) * (seeds_index + 1));

        token = strtok(NULL, " ");
    }

    free(mut_line);

    *length = seeds_index;
    return seeds;
}