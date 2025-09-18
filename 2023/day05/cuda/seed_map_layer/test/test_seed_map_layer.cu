#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map_layer.cuh"

#include <stdlib.h>

void test_init(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    seed_map_layer_term(seed_map_layer);
}

void test_add_seed_map(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    SEED_MAP seed_map = {
        .source = 1,
        .target = 5,
        .size = 2
    };

    seed_map_layer_add_map(seed_map_layer, &seed_map);

    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->source, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->target, 5);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->size, 2);

    seed_map_layer_term(seed_map_layer);
}

void test_flatten_seed_map_layer_1_map(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    SEED_MAP seed_map = {
        .source = 1,
        .target = 5,
        .size = 2
    };

    seed_map_layer_add_map(seed_map_layer, &seed_map);

    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->source, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->target, 5);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->size, 2);

    uint32_t flat_seed_map_layer_size = 0;
    uint8_t *flat_seed_map_layer = seed_map_layer_flatten(seed_map_layer, &flat_seed_map_layer_size);

    uint64_t source = (uint64_t)*flat_seed_map_layer;
    uint64_t target = (uint64_t)*(flat_seed_map_layer + 8);
    uint64_t size =   (uint64_t)*(flat_seed_map_layer + 16);

    CU_ASSERT_EQUAL(flat_seed_map_layer_size, sizeof(SEED_MAP));
    CU_ASSERT_EQUAL(source, 1);
    CU_ASSERT_EQUAL(target, 5);
    CU_ASSERT_EQUAL(size, 2);

    seed_map_layer_term(seed_map_layer);
    free(flat_seed_map_layer);
}

void test_unflatten_seed_map_layer_1_map(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    SEED_MAP seed_map = {
        .source = 1,
        .target = 5,
        .size = 2
    };

    seed_map_layer_add_map(seed_map_layer, &seed_map);

    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->source, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->target, 5);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->size, 2);

    uint32_t flat_seed_map_layer_size = 0;
    uint8_t *flat_seed_map_layer = seed_map_layer_flatten(seed_map_layer, &flat_seed_map_layer_size);

    uint64_t source = (uint64_t)*flat_seed_map_layer;
    uint64_t target = (uint64_t)*(flat_seed_map_layer + 8);
    uint64_t size =   (uint64_t)*(flat_seed_map_layer + 16);

    CU_ASSERT_EQUAL(flat_seed_map_layer_size, sizeof(SEED_MAP));
    CU_ASSERT_EQUAL(source, 1);
    CU_ASSERT_EQUAL(target, 5);
    CU_ASSERT_EQUAL(size, 2);

    SEED_MAP_LAYER *unflattened = seed_map_layer_unflatten(flat_seed_map_layer, flat_seed_map_layer_size);

    CU_ASSERT_EQUAL(unflattened->num_seed_maps, 1);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].source, 1);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].target, 5);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].size, 2);

    seed_map_layer_term(unflattened);
    seed_map_layer_term(seed_map_layer);
    free(flat_seed_map_layer);
}

void test_flatten_seed_map_layer_2_maps(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    SEED_MAP seed_map_1 = {
        .source = 1,
        .target = 5,
        .size = 2
    };

    SEED_MAP seed_map_2 = {
        .source = 10,
        .target = 50,
        .size = 5
    };

    seed_map_layer_add_map(seed_map_layer, &seed_map_1);
    seed_map_layer_add_map(seed_map_layer, &seed_map_2);

    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 2);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->source, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->target, 5);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->size, 2);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->source, 10);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->target, 50);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->size, 5);

    uint32_t flat_seed_map_layer_size = 0;
    uint8_t *flat_seed_map_layer = seed_map_layer_flatten(seed_map_layer, &flat_seed_map_layer_size);

    uint64_t source_1 = (uint64_t)*flat_seed_map_layer;
    uint64_t target_1 = (uint64_t)*(flat_seed_map_layer + 8);
    uint64_t size_1 =   (uint64_t)*(flat_seed_map_layer + 16);

    uint64_t source_2 = (uint64_t)*(flat_seed_map_layer + 24);
    uint64_t target_2 = (uint64_t)*(flat_seed_map_layer + 32);
    uint64_t size_2 =   (uint64_t)*(flat_seed_map_layer + 40);

    CU_ASSERT_EQUAL(flat_seed_map_layer_size, sizeof(SEED_MAP) * 2);
    CU_ASSERT_EQUAL(source_1, 1);
    CU_ASSERT_EQUAL(target_1, 5);
    CU_ASSERT_EQUAL(size_1, 2);
    CU_ASSERT_EQUAL(source_2, 10);
    CU_ASSERT_EQUAL(target_2, 50);
    CU_ASSERT_EQUAL(size_2, 5);

    seed_map_layer_term(seed_map_layer);
    free(flat_seed_map_layer);
}

void test_unflatten_seed_map_layer_2_maps(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);
    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 0);

    SEED_MAP seed_map_1 = {
        .source = 1,
        .target = 5,
        .size = 2
    };

    SEED_MAP seed_map_2 = {
        .source = 10,
        .target = 50,
        .size = 5
    };

    seed_map_layer_add_map(seed_map_layer, &seed_map_1);
    seed_map_layer_add_map(seed_map_layer, &seed_map_2);

    CU_ASSERT_EQUAL(seed_map_layer->num_seed_maps, 2);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->source, 1);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->target, 5);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps)->size, 2);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->source, 10);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->target, 50);
    CU_ASSERT_EQUAL((seed_map_layer->seed_maps + 1)->size, 5);

    uint32_t flat_seed_map_layer_size = 0;
    uint8_t *flat_seed_map_layer = seed_map_layer_flatten(seed_map_layer, &flat_seed_map_layer_size);

    uint64_t source_1 = (uint64_t)*flat_seed_map_layer;
    uint64_t target_1 = (uint64_t)*(flat_seed_map_layer + 8);
    uint64_t size_1 =   (uint64_t)*(flat_seed_map_layer + 16);

    uint64_t source_2 = (uint64_t)*(flat_seed_map_layer + 24);
    uint64_t target_2 = (uint64_t)*(flat_seed_map_layer + 32);
    uint64_t size_2 =   (uint64_t)*(flat_seed_map_layer + 40);

    CU_ASSERT_EQUAL(flat_seed_map_layer_size, sizeof(SEED_MAP) * 2);
    CU_ASSERT_EQUAL(source_1, 1);
    CU_ASSERT_EQUAL(target_1, 5);
    CU_ASSERT_EQUAL(size_1, 2);
    CU_ASSERT_EQUAL(source_2, 10);
    CU_ASSERT_EQUAL(target_2, 50);
    CU_ASSERT_EQUAL(size_2, 5);

    SEED_MAP_LAYER *unflattened = seed_map_layer_unflatten(flat_seed_map_layer, flat_seed_map_layer_size);

    CU_ASSERT_EQUAL(unflattened->num_seed_maps, 2);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].source, 1);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].target, 5);
    CU_ASSERT_EQUAL(unflattened->seed_maps[0].size, 2);
    CU_ASSERT_EQUAL(unflattened->seed_maps[1].source, 10);
    CU_ASSERT_EQUAL(unflattened->seed_maps[1].target, 50);
    CU_ASSERT_EQUAL(unflattened->seed_maps[1].size, 5);

    seed_map_layer_term(unflattened);
    seed_map_layer_term(seed_map_layer);
    free(flat_seed_map_layer);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapLayerTests", 0, 0);
    CU_add_test(suite, "Test init map layer", test_init);
    CU_add_test(suite, "Test add map", test_add_seed_map);
    CU_add_test(suite, "Test flatten map layer 1 map", test_flatten_seed_map_layer_1_map);
    CU_add_test(suite, "Test unflatten map layer 1 map", test_unflatten_seed_map_layer_1_map);
    CU_add_test(suite, "Test flatten map layer 2 maps", test_flatten_seed_map_layer_2_maps);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}