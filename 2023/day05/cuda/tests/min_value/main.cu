#include <stdio.h>
#include <cuda.h>

#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

const uint32_t block_count = 10;
const uint32_t thread_count = 10;

#define ARRAY_SIZE 1000000

__global__ void dkernel()
{ //__global__ indicate it is not normal kernel function but for GPU
    const uint32_t blockId = blockIdx.x;
    const uint32_t threadId = threadIdx.x;

    printf("Block Id: %u, Thread Id: %u\n", blockId, threadId);
}

int main ()
{
    uint32_t random_array[ARRAY_SIZE];

    uint32_t min_value = UINT32_MAX;
    for(uint32_t index = 0; index < ARRAY_SIZE; index++)
    {
        random_array[index] = rand();
    }

    for(uint32_t index = 0; index < ARRAY_SIZE; index++)
    {
        if(random_array[index] < min_value)
        {
            min_value = random_array[index];
        }
    }

    printf("Min Value %u\n", min_value);

    // dkernel <<<10, 10>>>();//<<<no. of blocks,no. of threads in in block>>>

    // cudaDeviceSynchronize(); //Tells GPU to do all work than synchronize GPU buffer with CPU.

    return 0;

}