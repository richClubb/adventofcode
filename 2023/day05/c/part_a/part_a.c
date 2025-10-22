#include "part_a.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <regex.h>
#include <limits.h>

#include "config.h"
#include "seed_map_layer.h"
#include "utils.h"

uint64_t part_a(const CONFIG *config)
{
    FILE *input_file = fopen(config->input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    uint64_t *seeds;
    uint64_t num_seeds;

    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers = (SEED_MAP_LAYER **)calloc(1, sizeof(SEED_MAP_LAYER *));
    uint64_t seed_map_layers_index = 0;

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
            seeds = get_seeds(line, &num_seeds);
            
            assert(seeds != NULL);
            
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

    uint64_t curr_seed_value = 0;
    uint64_t curr_seed_min = ULONG_MAX;
    for (uint64_t seed_index = 0; seed_index < num_seeds; seed_index++)
    {
        curr_seed_value = seeds[seed_index];

        for(uint64_t seed_map_layer_index = 0; seed_map_layer_index < seed_map_layers_index; seed_map_layer_index++)
        {
            SEED_MAP_LAYER *curr_seed_map_layer = seed_map_layers[seed_map_layer_index];
            seed_map_layer_map_seed(curr_seed_map_layer, &curr_seed_value);
        }

        if (curr_seed_value < curr_seed_min)
        {
            curr_seed_min = curr_seed_value;
        }
    }

    seed_map_layers_term(seed_map_layers, seed_map_layers_index);

    fclose(input_file);
    
    free(seeds);
    
    return curr_seed_min;
}