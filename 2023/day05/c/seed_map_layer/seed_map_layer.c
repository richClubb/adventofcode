#include "seed_map_layer.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "seed_map.h"

void seed_map_layer_init(SEED_MAP_LAYER *seed_map_layer)
{
    seed_map_layer->seed_map_count = 0;
    seed_map_layer->seed_maps = NULL;
}

void seed_map_layer_add_seed_map(SEED_MAP_LAYER *seed_map_layer, SEED_MAP *seed_map)
{
    SEED_MAP **new_seed_maps = (SEED_MAP **)malloc(sizeof(SEED_MAP *) * seed_map_layer->seed_map_count + 1);
    
    if( seed_map_layer->seed_maps != NULL )
    {
        memcpy(new_seed_maps, seed_map_layer->seed_maps, sizeof(SEED_MAP *) * seed_map_layer->seed_map_count);
        free(seed_map_layer->seed_maps);
    }

    new_seed_maps[seed_map_layer->seed_map_count] = seed_map;
    seed_map_layer->seed_map_count++;

    seed_map_layer->seed_maps = new_seed_maps;
}

bool seed_map_layer_map_seed(const SEED_MAP_LAYER *seed_map_layer, long *seed_value)
{
    // if this seed_map layer is null then fail
    assert(seed_map_layer != NULL);

    // if the seed map layer has no maps then fail out
    assert(seed_map_layer->seed_map_count != 0);

    for(
        unsigned int index = 0; 
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