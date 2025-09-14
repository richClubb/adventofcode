#ifndef __TEST_HEADER_PART_A_H__

#define __TEST_HEADER_PART_A_H__

#include "config.h"
#include "seed_map.h"

long *get_seeds(char *line, unsigned int *num_seeds);

SEED_MAP *get_seed_map(char *line);

char *get_seed_map_layer_name(char *line);

void part_a(const CONFIG *config);

#endif