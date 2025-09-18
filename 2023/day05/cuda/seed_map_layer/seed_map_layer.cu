#include "seed_map_layer.cuh"

#include <assert.h>

#include <stdlib.h>

#include "seed_map.cuh"

void seed_map_layer_init(SEED_MAP_LAYER **seed_map_layer)
{
    SEED_MAP_LAYER *temp;
    *seed_map_layer = (SEED_MAP_LAYER *)calloc(1, sizeof(SEED_MAP_LAYER));
    temp = *seed_map_layer;
    temp->seed_maps = (SEED_MAP *)calloc(0, sizeof(SEED_MAP));
    temp->num_seed_maps = 0;
}

void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer)
{
    assert(seed_map_layer != NULL);

    free(seed_map_layer->seed_maps);

    seed_map_layer->num_seed_maps = 0;

    free(seed_map_layer);
}

void seed_map_layer_add_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map)
{
    assert(seed_map_layer != NULL);

    seed_map_layer->seed_maps = (SEED_MAP *)realloc(seed_map_layer->seed_maps, seed_map_layer->num_seed_maps + 2 * sizeof(SEED_MAP));
    memcpy((seed_map_layer->seed_maps + seed_map_layer->num_seed_maps), seed_map, sizeof(SEED_MAP));
    seed_map_layer->num_seed_maps += 1;
}

uint8_t *seed_map_layer_flatten(SEED_MAP_LAYER *seed_map_layer, uint32_t *size)
{
    uint8_t *result = (uint8_t *)calloc(seed_map_layer->num_seed_maps, sizeof(SEED_MAP));

    memcpy(result, seed_map_layer->seed_maps, seed_map_layer->num_seed_maps * sizeof(SEED_MAP));

    *size = seed_map_layer->num_seed_maps * sizeof(SEED_MAP);
    return result;
}

SEED_MAP_LAYER *seed_map_layer_unflatten(uint8_t *data, uint32_t size)
{
    uint32_t seed_map_layer_count = size / sizeof(SEED_MAP);
    assert(size % sizeof(SEED_MAP) == 0);
    
    SEED_MAP_LAYER *seed_map_layer = (SEED_MAP_LAYER *)calloc(1, sizeof(SEED_MAP_LAYER));
    seed_map_layer->seed_maps = (SEED_MAP *)calloc(seed_map_layer_count + 1, sizeof(SEED_MAP));
    memcpy(seed_map_layer->seed_maps, data, sizeof(uint8_t) * size);

    seed_map_layer->num_seed_maps = seed_map_layer_count;

    return seed_map_layer;
}

__device__ void seed_map_layer_map_seed(SEED_MAP_LAYER *seed_map_layer, uint32_t seed)
{

}