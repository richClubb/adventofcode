#include "part_a.h"

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#include "config.h"
#include "seed_map_layer.h"
#include "utils.h"

uint64_t part_a_openmp(const CONFIG *config)
{
    uint64_t *seeds;
    uint64_t num_seeds = 0;

    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers = 0;

    injest_file_part_a(config->input_file_path, &seeds, &num_seeds, &seed_map_layers, &num_seed_map_layers);

    uint64_t curr_seed_value = 0;
    
    uint64_t *results = (uint64_t *)calloc(num_seeds, sizeof(uint64_t));
    
    #pragma omp parallel for num_threads(28)
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