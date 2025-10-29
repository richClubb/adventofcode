#include "part_b.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "seed_range.h"
#include "seed_map.h"
#include "seed_map_layer.h"

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