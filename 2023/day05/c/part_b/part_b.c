#include "part_b.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <regex.h>
#include <limits.h>
#include <stdint.h>

#include "config.h"
#include "seed_map_layer.h"
#include "seed_range.h"
#include "utils.h"

void injest_file_part_b(
    const char *input_file_path, 
    SEED_RANGE **seed_ranges_ptr, uint64_t *num_seed_ranges,
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
            *seed_ranges_ptr = get_seed_ranges(line, num_seed_ranges);
            
            assert(*seed_ranges_ptr != NULL);
            
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

uint64_t part_b(const CONFIG *config)
{
    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges;

    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers = 0;

    injest_file_part_b(
        config->input_file_path, 
        &seed_ranges, &num_seed_ranges,
        &seed_map_layers, &num_seed_map_layers
    );

    uint64_t *results = (uint64_t *)calloc(num_seed_ranges, sizeof(uint64_t));

    // don't like this too much, think of refactoring it.
    for(uint64_t seed_ranges_index = 0; seed_ranges_index < num_seed_ranges; seed_ranges_index++)
    {
        uint64_t seed_range_start = seed_ranges[seed_ranges_index].start;
        uint64_t seed_range_end = seed_ranges[seed_ranges_index].start + seed_ranges[seed_ranges_index].size;
        uint64_t range_min = ULONG_MAX;

        for 
        (
            uint64_t seed_index = seed_range_start; 
            seed_index < seed_range_end; 
            seed_index++
        )
        {
            uint64_t curr_seed_value = seed_index;
            for
            (
                uint64_t seed_map_layer_index = 0; 
                seed_map_layer_index < num_seed_map_layers; 
                seed_map_layer_index++
            )
            {
                SEED_MAP_LAYER *curr_seed_map_layer = seed_map_layers[seed_map_layer_index];
                seed_map_layer_map_seed(curr_seed_map_layer, &curr_seed_value);
            }

            if (curr_seed_value < range_min) range_min = curr_seed_value;
        }

        results[seed_ranges_index] = range_min;
    }

    uint64_t curr_seed_min = UINT64_MAX;

    for(uint64_t index = 0; index < num_seed_ranges; index++)
    {
        if (results[index] < curr_seed_min) curr_seed_min = results[index];
    }

    free(results);

    free(seed_ranges);

    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    return curr_seed_min;
}

unsigned long part_b_parallel(const CONFIG *config)
{
    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges;

    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers = 0;

    injest_file_part_b(
        config->input_file_path, 
        &seed_ranges, &num_seed_ranges,
        &seed_map_layers, &num_seed_map_layers
    );

    sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);
    
    SEED_RANGE *new_seed_ranges = split_seed_ranges_by_number(seed_ranges, &num_seed_ranges, 28);
    free(seed_ranges);
    
    printf("Num ranges %lu\n", num_seed_ranges);
    uint64_t *results = (uint64_t *)calloc(num_seed_ranges, sizeof(uint64_t));

    // don't like this too much, think of refactoring it.
    #pragma omp parallel for num_threads(28)
    for(uint64_t seed_ranges_index = 0; seed_ranges_index < num_seed_ranges; seed_ranges_index++)
    {
        uint64_t seed_range_start = new_seed_ranges[seed_ranges_index].start;
        uint64_t seed_range_end = new_seed_ranges[seed_ranges_index].start + new_seed_ranges[seed_ranges_index].size;

        uint64_t range_min = UINT64_MAX;
        for 
        (
            uint64_t seed_index = seed_range_start; 
            seed_index < seed_range_end; 
            seed_index++
        )
        {
            uint64_t curr_seed_value = seed_index;
            for
            (
                uint64_t seed_map_layer_index = 0; 
                seed_map_layer_index < num_seed_map_layers; 
                seed_map_layer_index++
            )
            {
                SEED_MAP_LAYER *curr_seed_map_layer = seed_map_layers[seed_map_layer_index];
                seed_map_layer_map_seed(curr_seed_map_layer, &curr_seed_value);
            }
            
            if ( curr_seed_value < range_min ) range_min = curr_seed_value;
        }

        results[seed_ranges_index] = range_min;
    }

    uint64_t curr_seed_min = UINT64_MAX;

    for (uint64_t index = 0; index < num_seed_ranges; index++)
    {
        if (results[index] < curr_seed_min) curr_seed_min = results[index];
    }

    free(results);

    free(new_seed_ranges);

    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    return curr_seed_min;
}