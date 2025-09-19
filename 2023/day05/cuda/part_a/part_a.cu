#include "part_a.cuh"

#include <cuda.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "seed_map.cuh"
#include "seed_map_layer.cuh"
#include "utils.cuh"

const uint32_t block_count = 1;
const uint32_t thread_count = 10;

__global__ void part_a_kernel(
    uint8_t *flat_seed_map_layers,
    uint32_t flat_seed_map_layers_size,
    uint32_t *seed_map_layer_sizes,
    uint32_t num_seed_map_layers,
    uint64_t *seed_values,
    uint32_t num_seeds,
    uint64_t *result
)
{
    const uint32_t blockId = blockIdx.x;
    const uint32_t threadId = threadIdx.x;

    //uint64_t *flat_seed_map_layers_ptr;

    uint64_t seed_value = seed_values[threadId];
    result[threadId] = 0;

    if(seed_value == 0)
    {
        return;
    }

    printf("Initial value %lu\n", seed_value);
    uint32_t flat_seed_map_layers_index_ptr = 0;
    for(uint32_t index_1 = 0; index_1 < num_seed_map_layers; index_1++)
    {
        uint32_t seed_map_layer_size = seed_map_layer_sizes[index_1];
        uint32_t num_maps = seed_map_layer_size / sizeof(SEED_MAP);
        bool layer_match_found = false;
        
        for (uint32_t index_2 = 0; index_2 < num_maps; index_2++)
        {
            uint64_t map_offset = flat_seed_map_layers_index_ptr + (index_2 * sizeof(SEED_MAP));
            printf("map_offset %lu\n", map_offset);
            uint64_t *source = (uint64_t *)(flat_seed_map_layers + (map_offset));
            uint64_t *target = (uint64_t *)(flat_seed_map_layers + (map_offset + ( 1 * sizeof(uint64_t))));
            uint64_t *size =   (uint64_t *)(flat_seed_map_layers + (map_offset + ( 2 * sizeof(uint64_t))));

            printf("source %lu\n", *source);
            if (
                (seed_value >= *source) && 
                (seed_value < *source + *size)
            )
            {
                layer_match_found = true;
                seed_value = (seed_value - *source) + *target;
                break;
            }
        }
        flat_seed_map_layers_index_ptr += seed_map_layer_sizes[index_1];

        if (layer_match_found)
        {
            continue;
        }
    }

    printf("Final value: %lu\n", seed_value);
    result[threadId] = seed_value;
}


uint64_t part_a(const CONFIG *config)
{
    FILE *input_file = fopen(config->input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    uint64_t *seeds;
    unsigned int num_seeds;
    // check for memory usage
    SEED_MAP_LAYER **seed_map_layers = (SEED_MAP_LAYER **)calloc(1, sizeof(SEED_MAP_LAYER *));
    unsigned int num_seed_map_layers = 0;

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
            //memory check
            seed_map_layers = (SEED_MAP_LAYER **)realloc(seed_map_layers, sizeof(SEED_MAP_LAYER *) * (num_seed_map_layers + 2));

            seed_map_layers[num_seed_map_layers] = NULL;
            seed_map_layer_init(&seed_map_layers[num_seed_map_layers]);
            curr_layer = seed_map_layers[num_seed_map_layers];
            
            num_seed_map_layers += 1;
            
            continue;
        }

        //if nothing else then it must be a seed map line
        SEED_MAP *seed_map = get_seed_map(line);
        seed_map_layer_add_map(curr_layer, seed_map);
    }

    fclose(input_file);

    uint32_t flat_seed_map_layers_size = 0;
    uint8_t *flat_seed_map_layers = (uint8_t *)calloc(0, sizeof(uint8_t));

    uint32_t *map_layer_sizes = (uint32_t *)calloc(num_seed_map_layers, sizeof(uint32_t));
    for (uint32_t index = 0; index < num_seed_map_layers; index++)
    {
        uint32_t flat_map_layer_size = 0;
        uint8_t *flat_map_layer = seed_map_layer_flatten(seed_map_layers[index], &flat_map_layer_size);
        map_layer_sizes[index] = flat_map_layer_size;

        flat_seed_map_layers_size += flat_map_layer_size;
        flat_seed_map_layers = (uint8_t *)realloc(
            flat_seed_map_layers, 
            (flat_seed_map_layers_size * sizeof(uint8_t))
        );
        memcpy(flat_seed_map_layers + (flat_seed_map_layers_size - flat_map_layer_size), flat_map_layer, flat_map_layer_size);
    }

    uint8_t *gpu_seed_map_input;
    cudaMalloc(
        &gpu_seed_map_input, 
        flat_seed_map_layers_size * sizeof(uint8_t)
    );
    cudaMemcpy(
        gpu_seed_map_input, 
        flat_seed_map_layers, 
        flat_seed_map_layers_size * sizeof(uint8_t), 
        cudaMemcpyHostToDevice
    );

    uint32_t *gpu_seed_map_sizes;
    cudaMalloc(
        &gpu_seed_map_sizes, 
        flat_seed_map_layers_size * sizeof(uint8_t)
    );
    cudaMemcpy(
        gpu_seed_map_sizes, 
        map_layer_sizes, 
        num_seed_map_layers * sizeof(uint32_t), 
        cudaMemcpyHostToDevice
    );

    uint64_t min_value = UINT64_MAX;
    uint32_t seeds_remaining = num_seeds;
    uint32_t seed_index = 0;
    
    while(seeds_remaining)
    {
        uint64_t *batch_seeds = (uint64_t *)calloc(thread_count, sizeof(uint64_t));
        uint32_t batch_count = 0;

        if(seeds_remaining >= thread_count)
        {
            memcpy(batch_seeds, (seeds + seed_index), thread_count * sizeof(uint64_t));
            seeds_remaining -= thread_count;
            batch_count = thread_count;
            seed_index += thread_count;
        }
        else
        {
            memcpy(batch_seeds, (seeds + seed_index), seeds_remaining * sizeof(uint64_t));
            batch_count = seeds_remaining;
            seeds_remaining = 0;
        }

        uint64_t *gpu_seeds;
        cudaMalloc(
            &gpu_seeds, 
            thread_count * sizeof(uint64_t)
        );

        cudaMemcpy(
            gpu_seeds, 
            batch_seeds, 
            thread_count * sizeof(uint64_t), 
            cudaMemcpyHostToDevice
        );

        uint64_t *gpu_result;
        cudaMalloc(&gpu_result, batch_count * sizeof(uint64_t));

        part_a_kernel <<<block_count, thread_count>>>(
            gpu_seed_map_input,
            flat_seed_map_layers_size,
            gpu_seed_map_sizes, 
            num_seed_map_layers,
            gpu_seeds,
            batch_count,
            gpu_result
        );

        cudaDeviceSynchronize();

        uint64_t *results = (uint64_t *)calloc(thread_count, sizeof(uint64_t));
        cudaMemcpy(results, gpu_result, batch_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        for(uint32_t index = 0; index < batch_count; index++)
        {
            //printf("%lu\n", results[index]);
            if (results[index] < min_value)
            {
                min_value = results[index];
            }
        }
        
        free(results);
        free(batch_seeds);
        cudaFree(gpu_seeds);
        cudaFree(gpu_result);
    }

    seed_map_layers_term(seed_map_layers, num_seed_map_layers);

    free(seeds);
    
    return min_value;
}