#include <stdio.h>
#include <string.h>

#include "config.h"
#include "part_a.h"
#include "part_b.h"


int main(int argc, char **argv)
{
    printf("2023 - Day 5\n");

    CONFIG config;

    if(load_config(&config, argc, argv))
    {
        return 1;
    }

    switch(config.run_type)
    {
        case PART_A:
            printf("Running Part A\n");

            printf("Result is: %lu\n", part_a(&config));
            break;
        case PART_A_OPENMP:
            printf("Running Part A OpenCL\n");

            printf("Result is: %lu\n", part_a_openmp(&config));
            break;
        case PART_A_OPENCL:
            printf("Running Part A OpenCL\n");

            printf("Result is: %lu\n", part_a_opencl(&config));
            break;
        case PART_B:
            printf("Running Part B\n");

            printf("Result is: %lu\n", part_b(&config));
            break;
        case PART_B_OPENCL:
            printf("Running Part B OpenCL\n");

            printf("Result is: %lu\n", part_b_opencl(&config));
            break;
        case PART_B_OPENMP:
            printf("Running Part B OpenMP\n");

            printf("Result is: %lu\n", part_b_openmp(&config));
            break;
        default:
            printf("Unsupported run type");
            break;
    }

    return 0;
}