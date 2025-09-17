#ifndef __CONFIG_H__

#define __CONFIG_H__

#define LOAD_CONFIG_RTN_HELP 1
#define LOAD_CONFIG_FAIL 2

typedef struct config_t
{
    char input_file_path[512];
} CONFIG;

int load_config(CONFIG *config, int argc, char** argv);

#endif