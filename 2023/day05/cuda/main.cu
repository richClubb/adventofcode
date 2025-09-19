#include <stdio.h>
#include <cuda.h>

#include <stdint.h>
#include <limits.h>

#define BLOCK_NUM 10
#define THREAD_NUM 10

#include "seed_range.cuh"

__device__ void find_min_value(SEED_RANGE *seed_range, uint64_t *min_value)
{
    for(uint64_t index = seed_range->start; index < seed_range->start + seed_range->size; index++)
    {
        if ((index % 123456) < *min_value)
        {
            *min_value = (index);
        }
    }
}

__global__ void dkernel(SEED_RANGE *seed_ranges,  uint64_t *result)
{ //__global__ indicate it is not normal kernel function but for GPU
    const uint32_t blockId = blockIdx.x;
    const uint32_t threadId = threadIdx.x;

    SEED_RANGE *seed_range = (seed_ranges + blockId);

    uint64_t seed_range_size = seed_range->size;

    printf("start %lu, size %lu\n", seed_range->start, seed_range->size);

    uint64_t min_value = UINT64_MAX;
    find_min_value(seed_range, &min_value);
    result[blockId] = min_value;
}

int main ()
{ 
    SEED_RANGE *seed_ranges = (SEED_RANGE *)calloc(BLOCK_NUM, sizeof(SEED_RANGE));

    for(uint32_t index = 0; index < BLOCK_NUM; index++)
    {
        seed_ranges[index].start = (index + 1) * 10;
        seed_ranges[index].size = (index + 1) * 1000000;

        printf("Starting values %lu %lu\n", seed_ranges[index].start, seed_ranges[index].size);
    }

    SEED_RANGE *gpu_input;
    cudaMalloc(&gpu_input, BLOCK_NUM * sizeof(SEED_RANGE));
    cudaMemcpy(gpu_input, seed_ranges, BLOCK_NUM * sizeof(SEED_RANGE), cudaMemcpyHostToDevice);

    uint64_t *cpu_result = (uint64_t *)calloc(BLOCK_NUM, sizeof(uint64_t));
    uint64_t *gpu_result;
    cudaMalloc(&gpu_result, BLOCK_NUM * sizeof(uint64_t));

    dkernel <<<BLOCK_NUM, 1>>>(gpu_input, gpu_result);//<<<no. of blocks,no. of threads in in block>>>

    cudaMemcpy(cpu_result, gpu_result, BLOCK_NUM * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize(); //Tells GPU to do all work than synchronize GPU buffer with CPU.

    for(uint32_t index = 0; index < BLOCK_NUM; index++)
    {
        printf("Block %lu, result: %lu\n", index, *(cpu_result + index));
    }

    return 0;

}