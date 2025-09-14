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

    part_a(&config);

    part_b(&config);

    return result;
}