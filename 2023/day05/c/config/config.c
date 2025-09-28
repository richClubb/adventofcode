#include "config.h"

#include <stdio.h>

// Opt
#include <unistd.h>
#include <getopt.h>
#include <string.h>

int load_config(CONFIG *config, int argc, char** argv)
{
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
                memset(config->run_type, '\0', 512);
                strncpy(config->run_type, optarg, strlen(optarg));
                break;
            default:
                return LOAD_CONFIG_FAIL;
        }
    }

    return 0;
}