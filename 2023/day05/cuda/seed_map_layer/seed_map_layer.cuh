#ifndef __SEED_MAP_LAYER_H__

#define __SEED_MAP_LAYER_H__

#include "seed.cuh"
#include "seed_map.cuh"

typedef struct seed_map_layer_t{
    SEED_MAP *seed_maps;
    uint64_t num_seed_maps;
} SEED_MAP_LAYER;

SEED_MAP_LAYER *seed_map_layer_init();
void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer);
void seed_map_layer_add_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map);
void seed_map_layer_sort_maps(SEED_MAP_LAYER *seed_map_layer);
SEED seed_map_layer_map_seed(SEED_MAP_LAYER *seed_map_layer, SEED seed);

typedef struct seed_map_layers_t{
    SEED_MAP_LAYER **seed_map_layers;
    uint64_t num_seed_map_layers;
} SEED_MAP_LAYERS;

SEED_MAP_LAYERS *seed_map_layers_init();
void seed_map_layers_term(SEED_MAP_LAYERS *seed_map_layers);
void seed_map_layers_add_layer(SEED_MAP_LAYERS *seed_map_layers, SEED_MAP_LAYER *seed_map_layer);
SEED seed_map_layers_map_seed(SEED_MAP_LAYERS *seed_map_layers, SEED seed);

uint64_t *flatten_layers(SEED_MAP_LAYERS *seed_map_layers, uint64_t *seed_map_layers_sizes, uint64_t *total_size);

#endif