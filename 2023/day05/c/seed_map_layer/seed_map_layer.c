#include "seed_map_layer.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "seed_map.h"
#include "utils.h"

// inits memory and seed maps list with empty
void seed_map_layer_init(SEED_MAP_LAYER **seed_map_layer)
{
    assert(*seed_map_layer == NULL);

    *seed_map_layer = (SEED_MAP_LAYER *)calloc(1, sizeof(SEED_MAP_LAYER));

    SEED_MAP_LAYER *temp = *seed_map_layer;

    temp->seed_map_count = 0;
    temp->seed_maps = (SEED_MAP **)calloc(0, sizeof(SEED_MAP *));
}

void seed_map_layer_term(SEED_MAP_LAYER *seed_map_layer)
{
    for (unsigned int index = 0; index < seed_map_layer->seed_map_count; index++)
    {
        seed_map_term(seed_map_layer->seed_maps[index]);
        seed_map_layer->seed_maps[index] = NULL;
    }

    if(seed_map_layer->seed_maps)
    {
        free(seed_map_layer->seed_maps);
        seed_map_layer->seed_maps = NULL;
    }

    seed_map_layer->seed_map_count = 0;

    if(seed_map_layer->name)
    {
        free(seed_map_layer->name);
        seed_map_layer->name = NULL;
    }

    free(seed_map_layer);
    seed_map_layer = NULL;
}

void seed_map_layers_term(SEED_MAP_LAYER **seed_map_layers, uint64_t num_seed_map_layers)
{
    for(uint64_t index = 0; index < num_seed_map_layers; index++)
    {
        seed_map_layer_term(seed_map_layers[index]);
        seed_map_layers[index] = NULL;
    }
    free(seed_map_layers);
    seed_map_layers = NULL;
}

void seed_map_layer_add_seed_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map)
{
    seed_map_layer->seed_maps = (SEED_MAP **)realloc(seed_map_layer->seed_maps, sizeof(SEED_MAP *) * (seed_map_layer->seed_map_count + 1));

    seed_map_layer->seed_maps[seed_map_layer->seed_map_count] = seed_map;
    seed_map_layer->seed_map_count += 1;
}

bool seed_map_layer_map_seed(const SEED_MAP_LAYER *seed_map_layer, uint64_t *seed_value)
{
    // if this seed_map layer is null then fail
    assert(seed_map_layer != NULL);

    // if the seed map layer has no maps then fail out
    assert(seed_map_layer->seed_map_count != 0);

    for(
        uint64_t index = 0; 
        index < seed_map_layer->seed_map_count; 
        index++
    )
    {
        SEED_MAP *curr_map = seed_map_layer->seed_maps[index];
        if(seed_map_map_seed(curr_map, seed_value))
        {
            return true;
        }
    }

    return false;
}

void seed_map_layer_sort_seed_maps(SEED_MAP_LAYER *seed_map_layer)
{

}