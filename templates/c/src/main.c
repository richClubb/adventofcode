#include <stdio.h>

#include "config.h"

#include "part_a.h"
#include "part_b.h"

int main(int argc, char** argv) {

    printf("Advent of Code 2016 - Day 01\n");

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
        case PART_B:
            printf("Running Part B\n");

            printf("Result is: %lu\n", part_b(&config));
            break;
        default:
            printf("Unsupported run type");
            break;
    }
    
    return 0;
}