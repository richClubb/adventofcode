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

// this has a small 16 byte memory leak apparently but can't figure out where
uint64_t *seed_map_layer_flatten_layer(SEED_MAP_LAYER *seed_map_layer, uint64_t *size)
{
    
    uint64_t num_entries = seed_map_layer->seed_map_count * 3;
    uint64_t *result = (uint64_t *)calloc(num_entries, sizeof(uint64_t));

    uint64_t *index_ptr = result;
    for (uint64_t seed_maps_index = 0; seed_maps_index < seed_map_layer->seed_map_count; seed_maps_index++)
    {
        *(index_ptr++) = seed_map_layer->seed_maps[seed_maps_index]->source;
        *(index_ptr++) = seed_map_layer->seed_maps[seed_maps_index]->target;
        *(index_ptr++) = seed_map_layer->seed_maps[seed_maps_index]->size;  
    }

    *size = num_entries;
    return result;
}

uint64_t *seed_map_layer_flatten_layers(SEED_MAP_LAYER **seed_map_layers, uint64_t num_layers, uint64_t **layer_sizes, uint64_t *total_size)
{
    
    *layer_sizes = (uint64_t *)calloc(num_layers, sizeof(uint64_t));

    *total_size = 0;

    for (uint64_t sml_index = 0; sml_index < num_layers; sml_index++)
    {
        *total_size += (seed_map_layers[sml_index]->seed_map_count * 3);
    }

    uint64_t *flat_seed_map_layers = (uint64_t *)calloc(*total_size, sizeof(uint64_t));

    uint64_t *fsml_ptr = flat_seed_map_layers;
    for (uint64_t sml_index = 0; sml_index < num_layers; sml_index++)
    {  
        uint64_t *flat_layer = seed_map_layer_flatten_layer(seed_map_layers[sml_index], &(*layer_sizes)[sml_index]);
        uint64_t size = (*layer_sizes)[sml_index];
        
        memcpy(fsml_ptr, flat_layer, size * sizeof(uint64_t));
        
        fsml_ptr += size;

        free(flat_layer);
    }

    return flat_seed_map_layers;
}

uint64_t seed_map_layers_map_seed_flat_layers(uint64_t *seed_map_layer_sizes, uint64_t *flat_seed_map_layers, uint64_t num_seed_map_layers, uint64_t seed_value)
{
    
    uint64_t seed_val = seed_value;
    
    uint64_t *layer_ptr = flat_seed_map_layers;                                
    for (uint64_t sml_index = 0; sml_index < num_seed_map_layers; sml_index++) 
    {                                                                          
        uint64_t num_maps = seed_map_layer_sizes[sml_index];                   
        uint64_t *map_ptr = layer_ptr;                                         
        for (uint64_t m_index = 0; m_index < num_maps; m_index++)        
        {                                                                      
            uint64_t m_source = *(map_ptr);                                    
            uint64_t m_target = *(map_ptr + 1);                                
            uint64_t m_size = *(map_ptr + 2);                                  
                                                                               
            if ((seed_val >= m_source) && (seed_val < m_source + m_size))      
            {                                                                  
                seed_val = seed_val - m_source + m_target;                     
                break;                                                         
            }                                                                  
                                                                               
            map_ptr += 3;                                                      
        }                                                                      
                                                                               
        layer_ptr += (num_maps * 3);                                                 
    }

    return seed_val;
}