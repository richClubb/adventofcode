#include "part_a.h"

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "config.h"
#include "utils.h"
#include "seed_calc.h"

// manually interprets the line
SEED_RANGE *get_seed_ranges(const char *line, uint64_t *num_seed_ranges)
{
    // seed line always starts with 'seeds: ' so atrip that out
    uint64_t seeds_substring_len = strlen(line) - sizeof("seeds: ");
    char *seeds_substring = (char *)calloc(seeds_substring_len + 1, sizeof(char));

    // if this fails then we can't continue
    assert(seeds_substring != NULL);
    
    // copy in the substring
    strncpy(seeds_substring, line + 7, seeds_substring_len);

    // allocate the inital seeds array
    uint64_t num_numbers;
    uint64_t *numbers = extract_number_list(seeds_substring, &num_numbers);
    // free seeds_substring as we no longer need it.

    SEED_RANGE *seed_ranges = (SEED_RANGE *)calloc(num_numbers / 2, sizeof(SEED_RANGE));
    *num_seed_ranges = 0;
    for (uint64_t index = 0; index < num_numbers; index += 2)
    {
        seed_ranges[*num_seed_ranges].start = numbers[index];
        seed_ranges[*num_seed_ranges].size = numbers[index + 1];
        *num_seed_ranges += 1;
    }

    free(seeds_substring);

    return seed_ranges;
}

void injest_file_part_b(
    const char *input_file_path,
    SEED_RANGE **seed_ranges, uint64_t *num_seed_ranges,
    SEED_MAP_LAYERS *seed_map_layers
)
{
    FILE *input_file = fopen(input_file_path, "r");

    seed_map_layers->num_seed_map_layers = 0;
    seed_map_layers->seed_map_layers = (SEED_MAP_LAYER *)calloc(0, sizeof(SEED_MAP_LAYER));

    assert(input_file != NULL);

    SEED_MAP_LAYER *curr_seed_map_layer;
    for(
        char line[256]; 
        fgets(line, sizeof(line), input_file) != NULL;
    ) 
    {
        if ( strlen(line) == 1 )
        {
            continue;
        }

        if ( strstr(line, "seeds:") != NULL )
        {   
            *seed_ranges = get_seed_ranges(line, num_seed_ranges);
            continue;
        }

        if ( strstr(line, ":") != NULL )
        {
            seed_map_layers->num_seed_map_layers += 1;
            seed_map_layers->seed_map_layers = (SEED_MAP_LAYER *)realloc(seed_map_layers->seed_map_layers, seed_map_layers->num_seed_map_layers * sizeof(SEED_MAP_LAYER));
            curr_seed_map_layer = (seed_map_layers->seed_map_layers + (seed_map_layers->num_seed_map_layers - 1));
            curr_seed_map_layer->seed_maps = (SEED_MAP *)calloc(0, sizeof(SEED_MAP));
            curr_seed_map_layer->num_seed_maps = 0;
            continue;
        }

        curr_seed_map_layer->num_seed_maps += 1;
        curr_seed_map_layer->seed_maps = (SEED_MAP *)realloc(curr_seed_map_layer->seed_maps, curr_seed_map_layer->num_seed_maps * sizeof(SEED_MAP));

        uint64_t length = 0;
        uint64_t *map_values = extract_number_list(line, &length);
        SEED_MAP *curr_seed_map = (curr_seed_map_layer->seed_maps + (curr_seed_map_layer->num_seed_maps - 1));

        curr_seed_map->source = map_values[1];
        curr_seed_map->target = map_values[0];
        curr_seed_map->size   = map_values[2];
        free(map_values);
    }

    fclose(input_file);
}

uint64_t part_b(const CONFIG *config)
{

    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges;

    SEED_MAP_LAYERS seed_map_layers;

    injest_file_part_b(config->input_file_path, &seed_ranges, &num_seed_ranges, &seed_map_layers);

    uint64_t curr_seed_min = UINT64_MAX;

    for (uint64_t index = 0; index < num_seed_ranges; index++)
    {
        uint64_t seed_range_min = seed_range_find_min((seed_ranges + index), &seed_map_layers);
        if (seed_range_min < curr_seed_min) curr_seed_min = seed_range_min;
    }

    return curr_seed_min;
}