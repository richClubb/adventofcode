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
            // (*seed_map_layers_ptr)[seed_map_layers_index] = (SEED_MAP_LAYER *)calloc(1, sizeof(SEED_MAP_LAYER));
            // (*seed_map_layers_ptr)[seed_map_layers_index]->seed_map_count = seed_map_layers_index;

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

uint64_t part_a(const CONFIG *config)
{
    uint64_t *seeds;
    uint64_t num_seeds = 0;

    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers = 0;

    injest_file_part_a(config->input_file_path, &seeds, &num_seeds, &seed_map_layers, &num_seed_map_layers);

    uint64_t curr_seed_value = 0;
    
    uint64_t *results = (uint64_t *)calloc(num_seeds, sizeof(uint64_t));
    for(uint64_t seed_index = 0; seed_index < num_seeds; seed_index++)
    {
        uint64_t curr_value = seeds[seed_index];
        for(uint64_t index_1 = 0; index_1 < num_seed_map_layers; index_1++)
        {
            SEED_MAP_LAYER *curr_layer = seed_map_layers[index_1];
            seed_map_layer_map_seed(curr_layer, &curr_value);
        }
        results[seed_index] = curr_value;
    }

    free(seeds);
    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    uint64_t curr_seed_min = ULONG_MAX;
    for (uint64_t index = 0; index < num_seeds; index++)
    {
        if (results[index] < curr_seed_min) curr_seed_min = results[index];
    }

    free(results);
    
    return curr_seed_min;
}