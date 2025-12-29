#ifndef __SEED_CALC_H__

#define __SEED_CALC_H__

#include <stdint.h>
#include <stdbool.h>

typedef struct SeedRange_t {
    uint64_t start;
    uint64_t size;
} SEED_RANGE;

typedef struct SeedMap_t {
    uint64_t source;
    uint64_t target;
    uint64_t size;
} SEED_MAP;

typedef struct SeedMapLayer_t {
    uint64_t num_seed_maps;
    SEED_MAP *seed_maps;
} SEED_MAP_LAYER;

typedef struct SeedMapLayers_t {
    uint64_t num_seed_map_layers;
    SEED_MAP_LAYER *seed_map_layers;
} SEED_MAP_LAYERS;

// bool seed_map_map_seed(uint64_t *value, const SEED_MAP *seed_map);
// uint64_t seed_map_layer_map_seed(const uint64_t value, const SEED_MAP_LAYER *seed_map_layer);
uint64_t seed_map_layers_map_seed(const uint64_t value, const SEED_MAP_LAYERS *seed_map_layers);
uint64_t seed_range_find_min(const SEED_RANGE *seed_range, const SEED_MAP_LAYERS *seed_map_layers);

#endif