#ifndef __PART_B_H__

#define __PART_B_H__

#include <stdint.h>

#include "config.h"
#include "seed_range.h"
#include "seed_map_layer.h"

uint64_t part_b(const CONFIG *config);
uint64_t part_b_opencl(const CONFIG *config);
uint64_t part_b_openmp(const CONFIG *config);

void injest_file_part_b(
    const char *input_file_path, 
    SEED_RANGE **seed_ranges_ptr, uint64_t *num_seed_ranges,
    SEED_MAP_LAYER ***seed_map_layers_ptr, uint64_t *num_seed_map_layers
);

#endif 