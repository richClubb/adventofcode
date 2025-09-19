#include "config.cuh"

#include <stdio.h>

// Opt
#include <unistd.h>
#include <getopt.h>
#include <string.h>

int load_config(CONFIG *config, int argc, char** argv)
{
        for (
        int opt;
        (opt = getopt(argc, argv, "hi:")) != -1;)
    {
        switch (opt)
        {
            case 'h':
                printf("  -i [path] - input_file_path\n");
                return LOAD_CONFIG_RTN_HELP;
            case 'i':
                strncpy(config->input_file_path, optarg, strlen(optarg));
                break;
            default:
                return LOAD_CONFIG_FAIL;
        }
    }
}