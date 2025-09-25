#include "part_b.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <regex.h>
#include <limits.h>

#include "config.h"
#include "seed_map_layer.h"
#include "seed_range.h"
#include "utils.h"

unsigned long part_b(const CONFIG *config)
{
    FILE *input_file = fopen(config->input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    SEED_RANGE **seed_ranges;
    unsigned int num_seed_ranges;

    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers = (SEED_MAP_LAYER **)malloc(sizeof(SEED_MAP_LAYER *) * 1);
    unsigned int seed_map_layers_index = 0;

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
            seed_ranges = get_seed_ranges(line, &num_seed_ranges);
            continue;
        }

        if ( strstr(line, ":") != NULL )
        {
            // memory check
            seed_map_layers = (SEED_MAP_LAYER **)realloc(seed_map_layers, sizeof(SEED_MAP_LAYER *) * (seed_map_layers_index + 2));

            seed_map_layers[seed_map_layers_index] = NULL;
            seed_map_layer_init(&seed_map_layers[seed_map_layers_index]);
            curr_layer = seed_map_layers[seed_map_layers_index];

            curr_layer->name = (char *)calloc(strlen(line), sizeof(char));
            memcpy(curr_layer->name, line, strlen(line) - 2);
            
            seed_map_layers_index += 1;
            
            continue;
        }

        // if nothing else then it must be a seed map line
        SEED_MAP *seed_map = get_seed_map(line);
        seed_map_layer_add_seed_map(curr_layer, seed_map);
    }

    unsigned long curr_seed_min = ULONG_MAX;

    // don't like this too much, think of refactoring it.
    for(int seed_ranges_index = 0; seed_ranges_index < num_seed_ranges; seed_ranges_index++)
    {
        unsigned long seed_range_start = seed_ranges[seed_ranges_index]->start;
        unsigned long seed_range_end = seed_ranges[seed_ranges_index]->start + seed_ranges[seed_ranges_index]->size;
        unsigned long range_min = ULONG_MAX;

        for 
        (
            unsigned long seed_index = seed_range_start; 
            seed_index < seed_range_end; 
            seed_index++
        )
        {
            unsigned long curr_seed_value = seed_index;
            for
            (
                unsigned int seed_map_layer_index = 0; 
                seed_map_layer_index < seed_map_layers_index; 
                seed_map_layer_index++
            )
            {
                SEED_MAP_LAYER *curr_seed_map_layer = seed_map_layers[seed_map_layer_index];
                seed_map_layer_map_seed(curr_seed_map_layer, &curr_seed_value);
            }

            if (curr_seed_value < range_min)
            {
                range_min = curr_seed_value;
            }

            if (range_min < curr_seed_min)
            {
                curr_seed_min = range_min;
            }
        }
    }

    seed_ranges_term(seed_ranges, num_seed_ranges);

    seed_map_layers_term(seed_map_layers, seed_map_layers_index);

    fclose(input_file);

    return curr_seed_min;
}