#ifndef __CONFIG_H__

#define __CONFIG_H__

#define LOAD_CONFIG_RTN_HELP 1
#define LOAD_CONFIG_FAIL 2

enum RunType {
    PART_A = 0,
    PART_A_NON_KERNEL,
    PART_B,
    PART_B_NON_KERNEL,
};

typedef struct config_t
{
    char input_file_path[512];
    RunType run_type;
} CONFIG;

int load_config(CONFIG *config, int argc, char** argv);

#endif