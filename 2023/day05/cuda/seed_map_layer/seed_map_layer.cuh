#ifndef __SEED_MAP_LAYER_CUH__

#define __SEED_MAP_LAYER_CUH__

#include <stdint.h>

#include "seed_map.cuh"

typedef struct seed_map_layer_t
{
    SEED_MAP *seed_maps;
    uint32_t num_seed_maps;
} SEED_MAP_LAYER;

typedef struct seed_map_layers_t
{
    SEED_MAP *seed_map_layers;
    uint32_t num_seed_map_layers;
} SEED_MAP_LAYERS;

void seed_map_layer_init(SEED_MAP_LAYER **seed_map_layer);
void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer);

void seed_map_layers_term(SEED_MAP_LAYER **seed_map_layers, unsigned int num_seed_map_layers);

void seed_map_layer_add_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map);

uint8_t *seed_map_layer_flatten(SEED_MAP_LAYER *seed_map_layer, uint32_t *size);
SEED_MAP_LAYER *seed_map_layer_unflatten(uint8_t *data, uint32_t size);


__device__ void seed_map_layer_map_seed(SEED_MAP_LAYER *seed_map_layer, uint32_t seed);

#endif 