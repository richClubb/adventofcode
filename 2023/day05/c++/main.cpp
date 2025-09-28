
#include "config.h"

#include <stdio.h>
#include <string.h>

#include "part_a.h"
#include "part_b.h"

#define APP_FAIL 1

int main(int argc, char **argv)
{
    printf("2023 - Day 5\n");

    CONFIG config;

    int result = 0;

    result = load_config(&config, argc, argv);

    if(strcmp(config.run_type, "part_a") == 0)
    {
        printf("Running Part A\n");
        unsigned long part_a_min = part_a(config);

        printf("Part A: min value is '%lu'\n", part_a_min);
    }
    else if(strcmp(config.run_type, "part_b_ptr") == 0)
    {
        printf("Running Part B ptr version\n");
        unsigned long part_b_min = part_b_ptr_version(config);

        printf("Part B: min value is '%lu'\n", part_b_min);
    }
    else if(strcmp(config.run_type, "part_b_optional") == 0)
    {
        printf("Running Part B optional version\n");
        unsigned long part_b_min = part_b_optional_version(config);

        printf("Part B: min value is '%lu'\n", part_b_min);
    }
    else
    {
        printf("Invalid run\n");
        return 1;
    }

    return result;
}