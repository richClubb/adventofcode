#include <stdio.h>
#include <string.h>

#include "config.h"
#include "part_a.h"
#include "part_b.h"


int main(int argc, char **argv)
{
    printf("2023 - Day 5\n");

    CONFIG config;

    int result = 0;

    result = load_config(&config, argc, argv);

    if(strcmp(config.run_type, "part_a") == 0)
    {
        printf("Running Part A\n");
        unsigned long part_a_min = part_a(&config);

        printf("Part A: min value is '%lu'\n", part_a_min);
    }
    else if(strcmp(config.run_type, "part_b") == 0)
    {
        printf("Running Part B\n");
        unsigned long part_b_min = part_b(&config);

        printf("Part B: min value is '%lu'\n", part_b_min);
    }
    else if(strcmp(config.run_type, "part_b_parallel") == 0)
    {
        printf("Running Part B\n");
        unsigned long part_b_min = part_b_parallel(&config);

        printf("Part B: min value is '%lu'\n", part_b_min);
    }
    else
    {
        printf("Invalid run\n");
        return 1;
    }

    return result;
}