// /workspaces/adventofcode/2023/day05/cuda/part_a/manualtest/part_a_sample.txt

#include <assert.h>
#include <stdio.h>
#include <cuda.h>

#include "part_a.cuh"
#include "config.cuh"

int main(int argc, char **argv)
{

    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/cuda/part_a/manualtest/part_a_sample.txt"
    };
    
    uint64_t result = part_a(&config);
    //assert(result == 35);

    return 0;
}