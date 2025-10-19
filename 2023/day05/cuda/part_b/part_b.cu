#include "part_b.cuh"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "config.cuh"
#include "seed_range.cuh"
#include "seed_map_layer.cuh"

void injest_file(const char *file_path, SEED_RANGE **seed_ranges, uint64_t *num_seed_ranges, SEED_MAP_LAYERS *seed_map_layers)
{
    FILE *input_file = fopen(file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    SEED_MAP_LAYER *curr_layer;

    for(
        char line[256]; 
        fgets(line, sizeof(line), input_file) != NULL;
    ) 
    {
        if ( strlen(line) <= 1 )
        {
            // empty line skip
            continue;
        }

        if ( strstr(line, "seeds:") != NULL )
        {
            *seed_ranges = get_seed_ranges_from_line(line, num_seed_ranges);
            continue;
        }

        if ( strstr(line, ":") != NULL )
        {
            curr_layer = seed_map_layer_init();
            seed_map_layers_add_layer(seed_map_layers, curr_layer);
            continue;
        }

        SEED_MAP *curr_map = seed_map_from_string(line);
        seed_map_layer_add_map(curr_layer, curr_map);
        free(curr_map);
    }

    fclose(input_file);
}

// processes 1 seed
__global__ void part_b_kernel(
    uint64_t *seed_ranges,
    uint64_t *seed_map_layer_sizes,
    uint64_t num_seed_map_layers,
    uint64_t *flat_seed_map_layers,
    uint64_t flat_seed_map_layers_size,
    uint64_t *result
)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;

    SEED_RANGE *temp_seed_ranges = (SEED_RANGE *)seed_ranges;

    uint64_t seed_start = temp_seed_ranges[thread_id].start;
    uint64_t seed_end = temp_seed_ranges[thread_id].start + temp_seed_ranges[thread_id].size;

    uint64_t min_value = UINT64_MAX;
    for(uint64_t seed = seed_start; seed < seed_end; seed++)
    {
        uint64_t seed_value = seed;
        uint64_t layer_index = 0;
        for (uint64_t index_1 = 0; index_1 < num_seed_map_layers; index_1++)
        {
            uint64_t num_maps = seed_map_layer_sizes[index_1];
            for (uint64_t index_2 = 0; index_2 < num_maps; index_2++)
            {
                uint64_t map_index = (5 * index_2);
                uint64_t source_start = *(flat_seed_map_layers + layer_index + map_index + 0);
                uint64_t source_end = *(flat_seed_map_layers + layer_index + map_index + 1);
                uint64_t target_start = *(flat_seed_map_layers + layer_index + map_index + 2);

                if ((seed_value >= source_start) && (seed_value < source_end))
                {
                    seed_value = seed_value - source_start + target_start;
                    break;
                }
            }
            layer_index += num_maps * 5;
        }

        if (seed_value < min_value) min_value = seed_value;
    }

    result[thread_id] = min_value;
}

uint64_t part_b(const CONFIG* config)
{
    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges = 0;

    SEED_MAP_LAYERS *seed_map_layers = seed_map_layers_init();

    injest_file(config->input_file_path, &seed_ranges, &num_seed_ranges, seed_map_layers);

    sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);
    
    SEED_RANGE *new_seed_ranges = split_seed_ranges_by_number(seed_ranges, &num_seed_ranges, 1000);
    free(seed_ranges);

    uint64_t *gpu_seed_ranges;
    cudaMalloc(&gpu_seed_ranges, num_seed_ranges * sizeof(SEED_RANGE));

    uint64_t flat_layers_total_size = 0;
    uint64_t *seed_map_layers_sizes = (uint64_t *)calloc(seed_map_layers->num_seed_map_layers, sizeof(uint64_t));
    uint64_t *flattened_seed_map_layers = flatten_layers(seed_map_layers, seed_map_layers_sizes, &flat_layers_total_size);

    uint64_t *gpu_seed_map_layers_sizes;
    cudaMalloc(&gpu_seed_map_layers_sizes, seed_map_layers->num_seed_map_layers * sizeof(uint64_t));

    cudaMemcpy(
        gpu_seed_map_layers_sizes,
        seed_map_layers_sizes,
        seed_map_layers->num_seed_map_layers * sizeof(uint64_t),
        cudaMemcpyHostToDevice
    );

    uint64_t *gpu_flat_seed_map_layers;
    cudaMalloc(&gpu_flat_seed_map_layers, flat_layers_total_size * sizeof(uint64_t));

    cudaMemcpy(
        gpu_flat_seed_map_layers,
        flattened_seed_map_layers,
        flat_layers_total_size * sizeof(uint64_t),
        cudaMemcpyHostToDevice
    );

    uint64_t *gpu_result;
    cudaMalloc(&gpu_result, num_seed_ranges * sizeof(uint64_t));

    cudaMemcpy(
        gpu_seed_ranges, 
        new_seed_ranges, 
        num_seed_ranges * sizeof(SEED_RANGE), 
        cudaMemcpyHostToDevice
    );
    free(new_seed_ranges);

    part_b_kernel <<<1, num_seed_ranges>>>(
        gpu_seed_ranges,
        gpu_seed_map_layers_sizes,
        seed_map_layers->num_seed_map_layers,
        gpu_flat_seed_map_layers,
        flat_layers_total_size,
        gpu_result
    );

    cudaDeviceSynchronize();

    SEED min_value = UINT64_MAX;

    uint64_t *results = (uint64_t *)calloc(num_seed_ranges, sizeof(SEED));
    cudaMemcpy(results, gpu_result, num_seed_ranges * sizeof(SEED), cudaMemcpyDeviceToHost);

    for(uint32_t index = 0; index < num_seed_ranges; index++)
    {
        if (results[index] < min_value) min_value = results[index];
    }

    seed_map_layers_term(seed_map_layers);
    
    return min_value;
}

uint64_t part_b_non_kernel(const CONFIG* config)
{
    SEED_RANGE *seed_ranges;
    uint64_t num_seed_ranges = 0;

    SEED_MAP_LAYERS *seed_map_layers = seed_map_layers_init();

    injest_file(config->input_file_path, &seed_ranges, &num_seed_ranges, seed_map_layers);

    SEED min_value = UINT64_MAX;
    for(uint64_t index = 0; index < num_seed_ranges; index++)
    {
        SEED_RANGE *curr_range = &seed_ranges[index];
        uint64_t start = curr_range->start;
        uint64_t end = curr_range->start + curr_range->size;
        for(uint64_t seed = start; seed < end; seed++)
        {
            uint64_t result = seed_map_layers_map_seed(seed_map_layers, seed);

            if (result < min_value) min_value = result;
        }
    }

    free(seed_ranges);
    seed_map_layers_term(seed_map_layers);
    
    return min_value;
}