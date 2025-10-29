#include "part_b.h"

#include <stdint.h>

#include "seed_range.h"
#include "seed_map_layer.h"

uint64_t part_b_openmp(const CONFIG *config)
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

    free(new_seed_ranges);
    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    uint64_t curr_seed_min = UINT64_MAX;

    for (uint64_t index = 0; index < num_seed_ranges; index++)
    {
        if (results[index] < curr_seed_min) curr_seed_min = results[index];
    }

    free(results);

    return curr_seed_min;
}