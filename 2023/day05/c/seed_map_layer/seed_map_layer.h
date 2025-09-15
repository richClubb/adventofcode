#ifndef __SEED_MAP_LAYER_H__

#define __SEED_MAP_LAYER_H__

#include "seed_map.h"

typedef struct seed_map_layer_t
{
    char *name;
    SEED_MAP **seed_maps;
    unsigned int seed_map_count;
} SEED_MAP_LAYER;

void seed_map_layer_init(SEED_MAP_LAYER **seed_map_layer);

void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer);

void seed_map_layers_term(SEED_MAP_LAYER **seed_map_layers, unsigned int num_seed_map_layers);

void seed_map_layer_add_seed_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map);

bool seed_map_layer_map_seed(const SEED_MAP_LAYER *seed_map_layer, unsigned long *seed_value);

void seed_map_layer_sort_seed_maps(SEED_MAP_LAYER *seed_map_layer);

#endif