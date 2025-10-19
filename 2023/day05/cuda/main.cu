#include <stdint.h>
#include <stdio.h>

#include "config.cuh"
#include "part_a.cuh"
#include "part_b.cuh"

int main(int argc, char **argv)
{

    printf("2023 - Day 5\n");

    CONFIG config;

    int result = 0;

    result = load_config(&config, argc, argv);
    if (result != 0)
    {
        return result;
    }

    switch (config.run_type)
    {
        case PART_A:
            printf("Part A result: %lu\n", part_a(&config));
            break;
        case PART_A_NON_KERNEL:
            printf("Part A result: %lu\n", part_a_non_kernel(&config));
            break;
        case PART_B:
            printf("Part B result: %lu\n", part_b(&config));
            break;
        case PART_B_NON_KERNEL:
            printf("Part B result: %lu\n", part_b_non_kernel(&config));
            break;
        default:
            printf("Unknown run %d\n", config.run_type);
            return 1;
    }

    return 0;
}