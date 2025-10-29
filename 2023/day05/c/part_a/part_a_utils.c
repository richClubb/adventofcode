#include "part_a.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "seed_map_layer.h"
#include "utils.h"

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

void injest_file_part_a(
    const char *input_file_path, 
    uint64_t **seeds_ptr, uint64_t *num_seeds,
    SEED_MAP_LAYER ***seed_map_layers_ptr, uint64_t *num_seed_map_layers
)
{
    FILE *input_file = fopen(input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    *seed_map_layers_ptr = (SEED_MAP_LAYER **)calloc(1, sizeof(SEED_MAP_LAYER *));
    uint64_t seed_map_layers_index = 0;

    SEED_MAP_LAYER *curr_layer = NULL;
    char line[256];
    for(
        char line[256]; 
        fgets(line, sizeof(line), input_file) != NULL;
    ) 
    {
        SEED_MAP_LAYER *curr_layer;
        if ( strlen(line) == 1 )
        {
            continue;
        }

        if ( strstr(line, "seeds:") != NULL )
        {
            *seeds_ptr = get_seeds(line, num_seeds);
            
            assert(*seeds_ptr != NULL);
            
            continue;
        }

        if ( strstr(line, ":") != NULL )
        {
            (*seed_map_layers_ptr) = (SEED_MAP_LAYER **)realloc(*seed_map_layers_ptr, sizeof(SEED_MAP_LAYER *) * (seed_map_layers_index + 2));

            (*seed_map_layers_ptr)[seed_map_layers_index] = NULL;
            (*seed_map_layers_ptr)[seed_map_layers_index + 1] = NULL;

            seed_map_layer_init(&(*seed_map_layers_ptr)[seed_map_layers_index]);

            curr_layer = (*seed_map_layers_ptr)[seed_map_layers_index];

            curr_layer->name = (char *)calloc(strlen(line), sizeof(char));
            memcpy(curr_layer->name, line, strlen(line) - 2);
            
            seed_map_layers_index++;
            continue;
        }

        seed_map_layer_add_seed_map(curr_layer, get_seed_map(line));
    }

    *num_seed_map_layers = seed_map_layers_index;
    fclose(input_file);
}