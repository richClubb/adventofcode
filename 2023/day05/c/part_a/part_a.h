#ifndef __PART_A_H__

#define __PART_A_H__

#include <stdint.h>

#include "config.h"
#include "seed_map_layer.h"

void injest_file_part_a(
    const char *input_file_path, 
    uint64_t **seeds_ptr, uint64_t *num_seeds,
    SEED_MAP_LAYER ***seed_map_layers_ptr, uint64_t *num_seed_map_layers
);


uint64_t part_a(const CONFIG *config);
uint64_t part_a_openmp(const CONFIG *config);
uint64_t part_a_opencl(const CONFIG *config);

#endif 