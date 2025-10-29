#ifndef __CONFIG_H__

#define __CONFIG_H__

#define LOAD_CONFIG_RTN_HELP 1
#define LOAD_CONFIG_FAIL 2

typedef enum RunType {
    PART_A,
    PART_A_OPENCL,
    PART_A_OPENMP,
    PART_B,
    PART_B_OPENCL,
    PART_B_OPENMP
} RUN_TYPE;

typedef struct config_t
{
    char input_file_path[512];
    RUN_TYPE run_type;
} CONFIG;

int load_config(CONFIG *config, int argc, char** argv);

#endif