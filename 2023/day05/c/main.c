#include <stdio.h>

#include "config.h"
#include "part_a.h"
#include "part_b.h"


int main(int argc, char **argv)
{
    printf("2023 - Day 5\n");

    CONFIG config;

    int result = 0;

    result = load_config(&config, argc, argv);

    unsigned long part_a_min = part_a(&config);

    printf("Part A: min value is '%d'\n", part_a_min);

    unsigned long part_b_min = part_b(&config);

    printf("Part B: min value is '%d'\n", part_b_min);

    return result;
}