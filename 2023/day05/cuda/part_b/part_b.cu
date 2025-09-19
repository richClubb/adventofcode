#include "part_b.cuh"

#include <cuda.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "seed_map.cuh"
#include "seed_map_layer.cuh"
#include "utils.cuh"
#include "seed_range.cuh"

const uint32_t block_count = 1;
const uint32_t thread_count = 2;

__global__ void part_b_kernel(
    uint8_t *flat_seed_map_layers,
    uint32_t flat_seed_map_layers_size,
    uint32_t *seed_map_layer_sizes,
    uint32_t num_seed_map_layers,
    uint64_t *seed_range_values,
    uint32_t num_seed_ranges,
    uint64_t *result
)
{
    const uint32_t blockId = blockIdx.x;
    const uint32_t threadId = threadIdx.x;

    uint64_t seed_range_start = seed_range_values[(threadId * 2)];
    uint64_t seed_range_end = seed_range_start + seed_range_values[(threadId * 2) + 1];
    result[threadId] = 0;

    uint64_t min_value = UINT64_MAX;
    for(uint64_t seed = seed_range_start; seed < seed_range_end; seed++)
    {
        uint64_t seed_value = seed;
        
        uint32_t flat_seed_map_layers_index_ptr = 0;
        for(uint32_t index_1 = 0; index_1 < num_seed_map_layers; index_1++)
        {
            uint32_t seed_map_layer_size = seed_map_layer_sizes[index_1];
            uint32_t num_maps = seed_map_layer_size / sizeof(SEED_MAP);
            bool layer_match_found = false;
            
            for (uint32_t index_2 = 0; index_2 < num_maps; index_2++)
            {
                uint64_t map_offset = flat_seed_map_layers_index_ptr + (index_2 * sizeof(SEED_MAP));
                uint64_t *source = (uint64_t *)(flat_seed_map_layers + (map_offset));
                uint64_t *target = (uint64_t *)(flat_seed_map_layers + (map_offset + ( 1 * sizeof(uint64_t))));
                uint64_t *size =   (uint64_t *)(flat_seed_map_layers + (map_offset + ( 2 * sizeof(uint64_t))));

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

        if(seed_value < min_value)
        {
            min_value = seed_value;
        }
    }

    result[threadId] = min_value;
}


uint64_t part_b(const CONFIG *config)
{
    FILE *input_file = fopen(config->input_file_path, "r");

    // bomb out if the file is NULL
    assert(input_file != NULL);

    SEED_RANGE **seed_ranges;
    uint32_t num_seed_ranges;
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
            seed_ranges = get_seed_ranges(line, &num_seed_ranges);
            
            assert(seed_ranges != NULL);
            
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
    
    uint64_t *buffer = (uint64_t *)calloc(num_seed_ranges, sizeof(SEED_RANGE));
    for(uint32_t index = 0; index < num_seed_ranges; index++)
    {
        buffer[(2*index)] = seed_ranges[index]->start;
        buffer[(2*index) + 1] = seed_ranges[index]->size;
    }

    uint64_t *gpu_seed_ranges;
    cudaMalloc(
        &gpu_seed_ranges, 
        thread_count * sizeof(SEED_RANGE)
    );

    cudaMemcpy(
        gpu_seed_ranges, 
        buffer, 
        thread_count * sizeof(SEED_RANGE), 
        cudaMemcpyHostToDevice
    );

    uint64_t *gpu_result;
    cudaMalloc(&gpu_result, thread_count * sizeof(uint64_t));

    part_b_kernel <<<block_count, thread_count>>>(
        gpu_seed_map_input,
        flat_seed_map_layers_size,
        gpu_seed_map_sizes, 
        num_seed_map_layers,
        gpu_seed_ranges,
        thread_count,
        gpu_result
    );

    cudaDeviceSynchronize();

    uint64_t *results = (uint64_t *)calloc(thread_count, sizeof(uint64_t));
    cudaMemcpy(results, gpu_result, thread_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for(uint32_t index = 0; index < thread_count; index++)
    {
        //printf("%lu\n", results[index]);
        if (results[index] < min_value)
        {
            min_value = results[index];
        }
    }
    
    free(results);
    cudaFree(gpu_seed_ranges);
    cudaFree(gpu_result);
    

    seed_map_layers_term(seed_map_layers, num_seed_map_layers);
    
    return min_value;
}