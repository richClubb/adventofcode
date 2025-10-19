#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map.cuh"
#include "seed_map_layer.cuh"

void test_seed_map_layer_init_term()
{
    SEED_MAP_LAYER *layer = seed_map_layer_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layer);
    CU_ASSERT_PTR_NOT_NULL(layer->seed_maps);
    CU_ASSERT_EQUAL(layer->num_seed_maps, 0);

    seed_map_layer_term(layer);
}

void test_seed_map_add_map()
{
    SEED_MAP_LAYER *layer = seed_map_layer_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

    {
        SEED_MAP seed_map = {
            .source_start = 1,
            .source_end = 4,
            .target_start = 2,
            .target_end = 5,
            .size = 3,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 1);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_end,   4);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_start, 2);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_end,   5);
    CU_ASSERT_EQUAL(layer->seed_maps[0].size,         3);

    {
        SEED_MAP seed_map = {
            .source_start = 5,
            .source_end = 10,
            .target_start = 20,
            .target_end = 25,
            .size = 5,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 1);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_end,   4);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_start, 2);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_end,   5);
    CU_ASSERT_EQUAL(layer->seed_maps[0].size,         3);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_start, 5);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_end,   10);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_start, 20);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_end,   25);
    CU_ASSERT_EQUAL(layer->seed_maps[1].size,         5);

    seed_map_layer_term(layer);
}

void test_seed_map_sort_maps()
{
    SEED_MAP_LAYER *layer = seed_map_layer_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

    {
        SEED_MAP seed_map = {
            .source_start = 5,
            .source_end = 10,
            .target_start = 20,
            .target_end = 25,
            .size = 5,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);

    {
        SEED_MAP seed_map = {
            .source_start = 1,
            .source_end = 4,
            .target_start = 2,
            .target_end = 5,
            .size = 3,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);

    CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 5);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_end,   10);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_start, 20);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_end,   25);
    CU_ASSERT_EQUAL(layer->seed_maps[0].size,         5);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_start, 1);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_end,   4);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_start, 2);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_end,   5);
    CU_ASSERT_EQUAL(layer->seed_maps[1].size,         3);

    seed_map_layer_sort_maps(layer);

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);

    CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 1);
    CU_ASSERT_EQUAL(layer->seed_maps[0].source_end,   4);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_start, 2);
    CU_ASSERT_EQUAL(layer->seed_maps[0].target_end,   5);
    CU_ASSERT_EQUAL(layer->seed_maps[0].size,         3);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_start, 5);
    CU_ASSERT_EQUAL(layer->seed_maps[1].source_end,   10);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_start, 20);
    CU_ASSERT_EQUAL(layer->seed_maps[1].target_end,   25);
    CU_ASSERT_EQUAL(layer->seed_maps[1].size,         5);

    seed_map_layer_term(layer);
}

void test_seed_map_map_seed()
{
    SEED_MAP_LAYER *layer = seed_map_layer_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

    {
        SEED_MAP seed_map = {
            .source_start = 5,
            .source_end = 10,
            .target_start = 20,
            .target_end = 25,
            .size = 5,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);

    {
        SEED_MAP seed_map = {
            .source_start = 1,
            .source_end = 4,
            .target_start = 2,
            .target_end = 5,
            .size = 3,
        };

        seed_map_layer_add_map(layer, &seed_map);
    }

    CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);

    {
        SEED result = seed_map_layer_map_seed(layer, 0);
        CU_ASSERT_EQUAL(result, 0);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 1);
        CU_ASSERT_EQUAL(result, 2);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 2);
        CU_ASSERT_EQUAL(result, 3);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 3);
        CU_ASSERT_EQUAL(result, 4);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 4);
        CU_ASSERT_EQUAL(result, 4);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 5);
        CU_ASSERT_EQUAL(result, 20);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 6);
        CU_ASSERT_EQUAL(result, 21);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 7);
        CU_ASSERT_EQUAL(result, 22);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 8);
        CU_ASSERT_EQUAL(result, 23);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 9);
        CU_ASSERT_EQUAL(result, 24);
    }

    {
        SEED result = seed_map_layer_map_seed(layer, 10);
        CU_ASSERT_EQUAL(result, 10);
    }
    seed_map_layer_term(layer);
}

void test_seed_map_layers_init_term()
{
    SEED_MAP_LAYERS *layers = seed_map_layers_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layers);
    CU_ASSERT_PTR_NOT_NULL(layers->seed_map_layers);
    CU_ASSERT_EQUAL(layers->num_seed_map_layers, 0);

    seed_map_layers_term(layers);
}

void test_seed_map_layers_add_layer()
{
    SEED_MAP_LAYERS *layers = seed_map_layers_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layers);
    CU_ASSERT_PTR_NOT_NULL(layers->seed_map_layers);
    CU_ASSERT_EQUAL(layers->num_seed_map_layers, 0);

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();

        CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

        {
            SEED_MAP seed_map = {
                .source_start = 5,
                .source_end = 10,
                .target_start = 20,
                .target_end = 25,
                .size = 5,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);
        CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 5);
        CU_ASSERT_EQUAL(layer->seed_maps[0].source_end, 10);
        CU_ASSERT_EQUAL(layer->seed_maps[0].size, 5);

        {
            SEED_MAP seed_map = {
                .source_start = 1,
                .source_end = 4,
                .target_start = 2,
                .target_end = 5,
                .size = 3,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);
        CU_ASSERT_EQUAL(layer->seed_maps[0].source_start, 5);
        CU_ASSERT_EQUAL(layer->seed_maps[0].source_end, 10);
        CU_ASSERT_EQUAL(layer->seed_maps[0].size, 5);
        CU_ASSERT_EQUAL(layer->seed_maps[1].source_start, 1);
        CU_ASSERT_EQUAL(layer->seed_maps[1].source_end, 4);
        CU_ASSERT_EQUAL(layer->seed_maps[1].size, 3);

        seed_map_layers_add_layer(layers, layer);
    }
    
    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 1);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].source_start, 5);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].source_start, 1);

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();

        CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

        {
            SEED_MAP seed_map = {
                .source_start = 50,
                .source_end = 55,
                .target_start = 30,
                .target_end = 35,
                .size = 5,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);

        {
            SEED_MAP seed_map = {
                .source_start = 10,
                .source_end = 13,
                .target_start = 20,
                .target_end = 23,
                .size = 3,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        seed_map_layers_add_layer(layers, layer);
    }

    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 2);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].source_start, 5);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].source_start, 1);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[0].source_start, 50);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[1].source_start, 10);

    seed_map_layers_term(layers);
}

void test_seed_map_layers_map_seed()
{
    SEED_MAP_LAYERS *layers = seed_map_layers_init();
    CU_ASSERT_PTR_NOT_NULL_FATAL(layers);
    CU_ASSERT_PTR_NOT_NULL(layers->seed_map_layers);
    CU_ASSERT_EQUAL(layers->num_seed_map_layers, 0);

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();

        CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

        {
            SEED_MAP seed_map = {
                .source_start = 5,
                .source_end = 10,
                .target_start = 20,
                .target_end = 25,
                .size = 5,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);

        {
            SEED_MAP seed_map = {
                .source_start = 1,
                .source_end = 4,
                .target_start = 2,
                .target_end = 5,
                .size = 3,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        seed_map_layers_add_layer(layers, layer);
    }

    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 1);

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();

        CU_ASSERT_PTR_NOT_NULL_FATAL(layer);

        {
            SEED_MAP seed_map = {
                .source_start = 50,
                .source_end = 55,
                .target_start = 30,
                .target_end = 35,
                .size = 5,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 1);

        {
            SEED_MAP seed_map = {
                .source_start = 10,
                .source_end = 13,
                .target_start = 20,
                .target_end = 23,
                .size = 3,
            };

            seed_map_layer_add_map(layer, &seed_map);
        }

        seed_map_layers_add_layer(layers, layer);
    }

    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 2);

    {
        SEED result = seed_map_layers_map_seed(layers, 0);
        CU_ASSERT_EQUAL(result, 0);
    }

    {
        SEED result = seed_map_layers_map_seed(layers, 1);
        CU_ASSERT_EQUAL(result, 2);
    }

    {
        SEED result = seed_map_layers_map_seed(layers, 50);
        CU_ASSERT_EQUAL(result, 30);
    }

    seed_map_layers_term(layers);
}

void test_add_map_to_layers()
{
    SEED_MAP_LAYERS *layers = seed_map_layers_init();

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();
        {
            SEED_MAP *map = seed_map_from_string("1 2 3\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("2 3 4\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);
        
        seed_map_layers_add_layer(layers, layer);

        CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 1);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].source_start, 2);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].target_start, 1);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].size, 3);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].source_start, 3);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].target_start, 2);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].size, 4);
    }

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();
        {
            SEED_MAP *map = seed_map_from_string("3 4 5\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("4 5 6\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("5 6 7\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 3);
        
        seed_map_layers_add_layer(layers, layer);      
    }
    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 2); 

    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].source_start, 2);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].target_start, 1);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].size, 3);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].source_start, 3);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].target_start, 2);
    CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].size, 4);

    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[0].source_start, 4);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[0].target_start, 3);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[0].size, 5);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[1].source_start, 5);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[1].target_start, 4);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[1].size, 6);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[2].source_start, 6);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[2].target_start, 5);
    CU_ASSERT_EQUAL(layers->seed_map_layers[1]->seed_maps[2].size, 7);
    
    
    seed_map_layers_term(layers);
}

void test_flatten_layers()
{
    SEED_MAP_LAYERS *layers = seed_map_layers_init();

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();
        {
            SEED_MAP *map = seed_map_from_string("1 2 3\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("2 3 4\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 2);
        
        seed_map_layers_add_layer(layers, layer);

        CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 1);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].source_start, 2);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].target_start, 1);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[0].size, 3);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].source_start, 3);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].target_start, 2);
        CU_ASSERT_EQUAL(layers->seed_map_layers[0]->seed_maps[1].size, 4);
    }

    {
        SEED_MAP_LAYER *layer = seed_map_layer_init();
        {
            SEED_MAP *map = seed_map_from_string("3 4 5\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("4 5 6\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        {
            SEED_MAP *map = seed_map_from_string("5 6 7\n");
            seed_map_layer_add_map(layer, map);
            free(map);
        }
        CU_ASSERT_EQUAL_FATAL(layer->num_seed_maps, 3);
        
        seed_map_layers_add_layer(layers, layer);      
    }
    CU_ASSERT_EQUAL_FATAL(layers->num_seed_map_layers, 2);
 
    uint64_t total_size;
    uint64_t *layers_num_maps = (uint64_t *)calloc(layers->num_seed_map_layers, sizeof(uint64_t));
    uint64_t *flat_layers = flatten_layers(layers, layers_num_maps, &total_size);

    CU_ASSERT_EQUAL(total_size, 25);

    CU_ASSERT_EQUAL(layers_num_maps[0], 2);
    CU_ASSERT_EQUAL(layers_num_maps[1], 3);

    CU_ASSERT_EQUAL(*(flat_layers + 0), 2);
    CU_ASSERT_EQUAL(*(flat_layers + 1), 5);
    CU_ASSERT_EQUAL(*(flat_layers + 2), 1);
    CU_ASSERT_EQUAL(*(flat_layers + 3), 4);
    CU_ASSERT_EQUAL(*(flat_layers + 4), 3);

    CU_ASSERT_EQUAL(*(flat_layers + 5), 3);
    CU_ASSERT_EQUAL(*(flat_layers + 6), 7);
    CU_ASSERT_EQUAL(*(flat_layers + 7), 2);
    CU_ASSERT_EQUAL(*(flat_layers + 8), 6);
    CU_ASSERT_EQUAL(*(flat_layers + 9), 4);

    CU_ASSERT_EQUAL(*(flat_layers + 10), 4);
    CU_ASSERT_EQUAL(*(flat_layers + 11), 9);
    CU_ASSERT_EQUAL(*(flat_layers + 12), 3);
    CU_ASSERT_EQUAL(*(flat_layers + 13), 8);
    CU_ASSERT_EQUAL(*(flat_layers + 14), 5);

    CU_ASSERT_EQUAL(*(flat_layers + 15), 5);
    CU_ASSERT_EQUAL(*(flat_layers + 16), 11);
    CU_ASSERT_EQUAL(*(flat_layers + 17), 4);
    CU_ASSERT_EQUAL(*(flat_layers + 18), 10);
    CU_ASSERT_EQUAL(*(flat_layers + 19), 6);

    CU_ASSERT_EQUAL(*(flat_layers + 20), 6);
    CU_ASSERT_EQUAL(*(flat_layers + 21), 13);
    CU_ASSERT_EQUAL(*(flat_layers + 22), 5);
    CU_ASSERT_EQUAL(*(flat_layers + 23), 12);
    CU_ASSERT_EQUAL(*(flat_layers + 24), 7);

    seed_map_layers_term(layers);
    free(layers_num_maps);
    free(flat_layers);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Seed Map Layer Tests", 0, 0);
    CU_add_test(suite, "", test_seed_map_layer_init_term);
    CU_add_test(suite, "", test_seed_map_add_map);
    CU_add_test(suite, "", test_seed_map_sort_maps);
    CU_add_test(suite, "", test_seed_map_map_seed);

    CU_add_test(suite, "", test_seed_map_layers_init_term);
    CU_add_test(suite, "", test_seed_map_layers_add_layer);
    CU_add_test(suite, "", test_seed_map_layers_map_seed);
    CU_add_test(suite, "", test_add_map_to_layers);

    CU_add_test(suite, "", test_flatten_layers);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}