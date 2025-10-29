#include "config.h"

#include <stdio.h>

// Opt
#include <unistd.h>
#include <getopt.h>
#include <string.h>



int load_config(CONFIG *config, int argc, char** argv)
{

    char run_type[512];
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
                memset(run_type, '\0', 512);
                strncpy(run_type, optarg, strlen(optarg));
                break;
            default:
                return LOAD_CONFIG_FAIL;
        }
    }

    if(strcmp(run_type, "part_a") == 0)
    {
        config->run_type = PART_A;
    }
    if(strcmp(run_type, "part_a_opencl") == 0)
    {
        config->run_type = PART_A_OPENCL;
    }
    else if(strcmp(run_type, "part_b") == 0)
    {
        config->run_type = PART_B;
    }
    else if(strcmp(run_type, "part_b_opencl") == 0)
    {
        config->run_type = PART_B_OPENCL;
    }
    else if(strcmp(run_type, "part_b_openmp") == 0)
    {
        config->run_type = PART_B_OPENMP;
    }
    // else
    // {
    //     printf("Invalid run\n");
    //     return 1;
    // }

    return 0;
}