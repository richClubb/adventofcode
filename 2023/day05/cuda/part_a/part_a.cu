#include "part_a.cuh"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "seed.cuh"
#include "seed_map_layer.cuh"

const uint32_t block_count = 1;

void injest_file(const char *file_path, SEED **seeds, uint64_t *num_seeds, SEED_MAP_LAYERS *seed_map_layers)
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
            *seeds = seed_get_seeds_from_line(line, num_seeds);
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

__global__ void part_a_kernel(
    uint64_t *seeds_flattened,
    uint64_t num_seeds,
    uint64_t *seed_map_layer_sizes,
    uint64_t num_seed_map_layers,
    uint64_t *flat_seed_map_layers,
    uint64_t flat_seed_map_layers_size,
    uint64_t *result
)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;

    SEED seed_value = seeds_flattened[thread_id];

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
    
    result[thread_id] = seed_value;
}

uint64_t part_a(const CONFIG* config)
{
    SEED *seeds;
    uint64_t num_seeds = 0;

    SEED_MAP_LAYERS *seed_map_layers = seed_map_layers_init();

    injest_file(config->input_file_path, &seeds, &num_seeds, seed_map_layers);

    uint64_t *gpu_seeds;
    cudaMalloc(&gpu_seeds, num_seeds * sizeof(SEED));

    cudaMemcpy(
        gpu_seeds, 
        seeds, 
        num_seeds * sizeof(SEED), 
        cudaMemcpyHostToDevice
    );
    free(seeds);

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
    free(seed_map_layers_sizes);

    uint64_t *gpu_flat_seed_map_layers;
    cudaMalloc(&gpu_flat_seed_map_layers, flat_layers_total_size * sizeof(uint64_t));

    cudaMemcpy(
        gpu_flat_seed_map_layers,
        flattened_seed_map_layers,
        flat_layers_total_size * sizeof(uint64_t),
        cudaMemcpyHostToDevice
    );
    free(flattened_seed_map_layers);

    uint64_t *gpu_result;
    cudaMalloc(&gpu_result, num_seeds * sizeof(SEED));

    part_a_kernel <<<block_count, num_seeds>>>(
        gpu_seeds,
        num_seeds,
        gpu_seed_map_layers_sizes,
        seed_map_layers->num_seed_map_layers,
        gpu_flat_seed_map_layers,
        flat_layers_total_size,
        gpu_result
    );

    cudaDeviceSynchronize();

    seed_map_layers_term(seed_map_layers);
    
    cudaFree(gpu_seeds);
    cudaFree(gpu_seed_map_layers_sizes);
    cudaFree(gpu_flat_seed_map_layers);

    SEED min_value = UINT64_MAX;

    uint64_t *results = (uint64_t *)calloc(num_seeds, sizeof(SEED));
    cudaMemcpy(results, gpu_result, num_seeds * sizeof(SEED), cudaMemcpyDeviceToHost);
    cudaFree(gpu_result);

    for(uint32_t index = 0; index < num_seeds; index++)
    {
        if (results[index] < min_value)
        {
            min_value = results[index];
        }
    }

    
    free(results);

    return min_value;
}

uint64_t part_a_non_kernel(const CONFIG* config)
{
    SEED *seeds;
    uint64_t num_seeds = 0;

    SEED_MAP_LAYERS *seed_map_layers = seed_map_layers_init();

    injest_file(config->input_file_path, &seeds, &num_seeds, seed_map_layers);

    SEED min_value = UINT64_MAX;
    for(uint64_t index = 0; index < num_seeds; index++)
    {
        uint64_t result = seed_map_layers_map_seed(seed_map_layers, seeds[index]);

        if (result < min_value) min_value = result;
    }

    free(seeds);
    seed_map_layers_term(seed_map_layers);
    
    return min_value;
}