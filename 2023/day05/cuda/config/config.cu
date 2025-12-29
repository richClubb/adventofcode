#include "config.cuh"

#include <stdio.h>

// Opt
#include <unistd.h>
#include <getopt.h>
#include <string.h>

int load_config(CONFIG *config, int argc, char** argv)
{

    char run_type_str[512];
    memset(run_type_str, '\0', 512);

    for 
    (
        int opt;
        (opt = getopt(argc, argv, "hi:r:")) != -1;
    )
    {
        switch (opt)
        {
            case 'h':
                printf("  -i [path] - input_file_path\n");
                return LOAD_CONFIG_RTN_HELP;
            case 'i':
                memset(config->input_file_path, '\0', 512);
                strncpy(config->input_file_path, optarg, strlen(optarg));
                break;
            case 'r':
                strncpy(run_type_str, optarg, strlen(optarg));
                break;
            default:
                return LOAD_CONFIG_FAIL;
        }
    }

    if ( strcmp(run_type_str, "part_a") == 0)
    {
        config->run_type = PART_A;
    }
    else if ( strcmp(run_type_str, "part_a_non_kernel") == 0)
    {
        config->run_type = PART_A_NON_KERNEL;
    }
    else if ( strcmp(run_type_str, "part_b") == 0)
    {
        config->run_type = PART_B;
    }
    else if ( strcmp(run_type_str, "part_b_non_kernel") == 0)
    {
        config->run_type = PART_B_NON_KERNEL;
    }
    else {
        printf("Unknown run type\n");
        return LOAD_CONFIG_FAIL;
    }

    return 0;
}