#include "seed_map_layer.cuh"

#include <stdio.h>

SEED_MAP_LAYER *seed_map_layer_init()
{
    SEED_MAP_LAYER *layer = (SEED_MAP_LAYER *)calloc(1, sizeof(SEED_MAP_LAYER));
    layer->seed_maps = (SEED_MAP *)calloc(0, sizeof(SEED_MAP));
    layer->num_seed_maps = 0;
    return layer;
}

void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer)
{
    if(seed_map_layer == NULL)
    {
        return;
    }

    if(seed_map_layer->seed_maps != NULL)
    {
        free(seed_map_layer->seed_maps);
    }

    seed_map_layer->num_seed_maps = 0;

    free(seed_map_layer);
}

void seed_map_layer_add_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map)
{
    const uint64_t index = seed_map_layer->num_seed_maps;
    ++seed_map_layer->num_seed_maps;

    seed_map_layer->seed_maps = (SEED_MAP *)realloc(seed_map_layer->seed_maps, seed_map_layer->num_seed_maps * sizeof(SEED_MAP));
    seed_map_layer->seed_maps[index] = *seed_map;
}

void seed_map_layer_sort_maps(SEED_MAP_LAYER *seed_map_layer)
{
    uint64_t num_maps = seed_map_layer->num_seed_maps;
    for (uint64_t index_1 = 0; index_1 < num_maps; index_1++)
    {
        for (uint64_t index_2 = 0; index_2 < num_maps - index_1 - 1; index_2++)
        {
            SEED_MAP seed_map_1 = seed_map_layer->seed_maps[index_2];
            SEED_MAP seed_map_2 = seed_map_layer->seed_maps[index_2 + 1];
            if(seed_map_1.source_start > seed_map_2.source_start)
            {
                SEED_MAP seed_map_temp = seed_map_1;
                seed_map_layer->seed_maps[index_2] = seed_map_2;
                seed_map_layer->seed_maps[index_2 + 1] = seed_map_temp;
            }
        }
    }
}

SEED seed_map_layer_map_seed(SEED_MAP_LAYER *seed_map_layer, SEED seed)
{
    SEED seed_val = seed;
    for (uint64_t index = 0; index < seed_map_layer->num_seed_maps; index++)
    {
        SEED_MAP *curr_map = &seed_map_layer->seed_maps[index];
        if(seed_map_map_seed(curr_map, &seed_val))
        {
            return seed_val;
        }
    }

    return seed_val;
}

SEED_MAP_LAYERS *seed_map_layers_init()
{
    SEED_MAP_LAYERS *layers = (SEED_MAP_LAYERS *)calloc(1, sizeof(SEED_MAP_LAYERS));
    layers->seed_map_layers = (SEED_MAP_LAYER **)calloc(0, sizeof(SEED_MAP_LAYER *));
    layers->num_seed_map_layers = 0;
    return layers;
}

void seed_map_layers_term(SEED_MAP_LAYERS *seed_map_layers)
{
    if(seed_map_layers == NULL)
    {
        return;
    }

    if(seed_map_layers->seed_map_layers != NULL)
    {
        for (uint64_t index = 0; index < seed_map_layers->num_seed_map_layers; index++)
        {
            seed_map_layer_term(seed_map_layers->seed_map_layers[index]);
        }

        free(seed_map_layers->seed_map_layers);
    }

    seed_map_layers->num_seed_map_layers = 0;

    free(seed_map_layers);
}

void seed_map_layers_add_layer(SEED_MAP_LAYERS *seed_map_layers, SEED_MAP_LAYER *seed_map_layer)
{
    const uint64_t index = seed_map_layers->num_seed_map_layers;
    ++seed_map_layers->num_seed_map_layers;

    seed_map_layers->seed_map_layers = (SEED_MAP_LAYER **)realloc(seed_map_layers->seed_map_layers, seed_map_layers->num_seed_map_layers * sizeof(SEED_MAP_LAYER *));
    seed_map_layers->seed_map_layers[index] = seed_map_layer;
}

SEED seed_map_layers_map_seed(SEED_MAP_LAYERS *seed_map_layers, SEED seed)
{
    SEED seed_val = seed;
    for( uint64_t index = 0; index < seed_map_layers->num_seed_map_layers; index++)
    {
        SEED_MAP_LAYER *curr_layer = seed_map_layers->seed_map_layers[index];
        seed_val = seed_map_layer_map_seed(curr_layer, seed_val);
    }

    return seed_val;
}

uint64_t *flatten_layers(SEED_MAP_LAYERS *seed_map_layers, uint64_t *seed_map_layers_sizes, uint64_t *total_size)
{
    uint64_t total_map_count = 0;
    for(uint64_t index_1 = 0; index_1 < seed_map_layers->num_seed_map_layers; index_1++)
    {
        uint64_t num_maps = seed_map_layers->seed_map_layers[index_1]->num_seed_maps;
        seed_map_layers_sizes[index_1] = num_maps;
        total_map_count += num_maps;
    }

    *total_size = total_map_count * 5;

    uint64_t *flat_layers = (uint64_t *)calloc(*total_size, sizeof(uint64_t));

    uint64_t overall_index = 0;
    for(uint64_t index_1 = 0; index_1 < seed_map_layers->num_seed_map_layers; index_1++)
    {
        SEED_MAP_LAYER *curr_layer = seed_map_layers->seed_map_layers[index_1];
        memcpy((flat_layers + overall_index), curr_layer->seed_maps, curr_layer->num_seed_maps * 5 * sizeof(uint64_t));
    
        overall_index += (curr_layer->num_seed_maps) * 5;
    }

    return flat_layers;
}