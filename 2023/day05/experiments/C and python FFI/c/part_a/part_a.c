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
    uint64_t **seeds, uint64_t *num_seeds,
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
            *seeds = get_seeds(line, num_seeds);
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

uint64_t part_a(const CONFIG *config)
{

    uint64_t *seeds;
    uint64_t num_seeds;

    SEED_MAP_LAYERS seed_map_layers;

    injest_file_part_a(config->input_file_path, &seeds, &num_seeds, &seed_map_layers);

    uint64_t curr_seed_min = UINT64_MAX;

    for (uint64_t index = 0; index < num_seeds; index++)
    {
        uint64_t final_value = seed_map_layers_map_seed(seeds[index], &seed_map_layers);

        if (final_value < curr_seed_min) curr_seed_min = final_value;
    }

    for(uint64_t seed_map_layer_index = 0; seed_map_layer_index < seed_map_layers.num_seed_map_layers; seed_map_layer_index++)
    {
        free(seed_map_layers.seed_map_layers[seed_map_layer_index].seed_maps);
    }

    free(seed_map_layers.seed_map_layers);
    free(seeds);

    return curr_seed_min;
}