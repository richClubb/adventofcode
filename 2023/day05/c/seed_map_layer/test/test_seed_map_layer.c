#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map_layer.h"

#include <stdlib.h>

void test_init(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;
    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_EQUAL_FATAL(seed_map_layer->seed_map_count, 0);
    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map_layer->seed_maps);

    seed_map_layer_term(seed_map_layer);
}

void test_add_1_map(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP *seed_map_1 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_1->source = 1;
    seed_map_1->target = 10;
    seed_map_1->size   = 5;

    seed_map_layer_add_seed_map(seed_map_layer, seed_map_1);

    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->source, 1);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->target, 10);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->size, 5);

    seed_map_layer_term(seed_map_layer);
}

void test_add_2_maps(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP *seed_map_1 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_1->source = 1;
    seed_map_1->target = 10;
    seed_map_1->size   = 5;

    SEED_MAP *seed_map_2 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_2->source = 20;
    seed_map_2->target = 7;
    seed_map_2->size   = 2;

    seed_map_layer_add_seed_map(seed_map_layer, seed_map_1);
    seed_map_layer_add_seed_map(seed_map_layer, seed_map_2);

    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->source, 1);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->target, 10);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[0]->size, 5);

    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[1]->source, 20);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[1]->target, 7);
    CU_ASSERT_EQUAL(seed_map_layer->seed_maps[1]->size, 2);

    seed_map_layer_term(seed_map_layer);
}

void test_basic_map_seed(void)
{
    SEED_MAP_LAYER *seed_map_layer = NULL;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP *seed_map_1 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_1->source = 1;
    seed_map_1->target = 10;
    seed_map_1->size   = 5;

    SEED_MAP *seed_map_2 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_2->source = 15;
    seed_map_2->target = 5;
    seed_map_2->size   = 2;

    seed_map_layer_add_seed_map(seed_map_layer, seed_map_1);
    seed_map_layer_add_seed_map(seed_map_layer, seed_map_2);

    unsigned long value = 0;
    CU_ASSERT_FALSE(seed_map_layer_map_seed(seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 0);

    value = 1;
    CU_ASSERT_TRUE(seed_map_layer_map_seed(seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 10);

    value = 6;
    CU_ASSERT_FALSE(seed_map_layer_map_seed(seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 6);

    value = 15;
    CU_ASSERT_TRUE(seed_map_layer_map_seed(seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 5);

    value = 16;
    CU_ASSERT_TRUE(seed_map_layer_map_seed(seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 6);

    seed_map_layer_term(seed_map_layer);
}

void test_sample_multi_layer(void)
{
    SEED_MAP_LAYER *seed_map_layer_1 = NULL;
    SEED_MAP_LAYER *seed_map_layer_2 = NULL;

    seed_map_layer_init(&seed_map_layer_1);
    seed_map_layer_init(&seed_map_layer_2);

    SEED_MAP *seed_map_1 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_1->source = 98;
    seed_map_1->target = 50;
    seed_map_1->size   = 2;

    SEED_MAP *seed_map_2 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_2->source = 50;
    seed_map_2->target = 52;
    seed_map_2->size   = 48;

    seed_map_layer_add_seed_map(seed_map_layer_1, seed_map_1);
    seed_map_layer_add_seed_map(seed_map_layer_1, seed_map_2);

    SEED_MAP *seed_map_3 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_3->source = 15;
    seed_map_3->target = 0;
    seed_map_3->size   = 37;

    SEED_MAP *seed_map_4 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_4->source = 52;
    seed_map_4->target = 37;
    seed_map_4->size   = 2;

    SEED_MAP *seed_map_5 = (SEED_MAP *)calloc(1, sizeof(SEED_MAP));
    seed_map_5->source = 0;
    seed_map_5->target = 39;
    seed_map_5->size   = 15;

    seed_map_layer_add_seed_map(seed_map_layer_2, seed_map_3);
    seed_map_layer_add_seed_map(seed_map_layer_2, seed_map_4);
    seed_map_layer_add_seed_map(seed_map_layer_2, seed_map_5);

    unsigned long seed_value = 79;

    CU_ASSERT_TRUE(seed_map_layer_map_seed(seed_map_layer_1, &seed_value));
    CU_ASSERT_EQUAL(seed_value, 81);

    CU_ASSERT_FALSE(seed_map_layer_map_seed(seed_map_layer_2, &seed_value));
    CU_ASSERT_EQUAL(seed_value, 81);

    seed_map_layer_term(seed_map_layer_1);
    seed_map_layer_term(seed_map_layer_2);
}

void test_seed_map_layers_term(void)
{
    SEED_MAP_LAYER **seed_map_layers = (SEED_MAP_LAYER **)calloc(2, sizeof(SEED_MAP_LAYER *));

    SEED_MAP_LAYER *seed_map_layer_1 = NULL;
    SEED_MAP_LAYER *seed_map_layer_2 = NULL;

    seed_map_layer_init(&seed_map_layer_1);
    seed_map_layer_init(&seed_map_layer_2);

    seed_map_layers[0] = seed_map_layer_1;
    seed_map_layers[1] = seed_map_layer_2;

    seed_map_layers_term(seed_map_layers, 2);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapLayerTests", 0, 0);
    CU_add_test(suite, "Test init map layer", test_init);
    CU_add_test(suite, "Test add 1 map to layer", test_add_1_map);
    CU_add_test(suite, "Test add 2 maps to layer", test_add_2_maps);
    CU_add_test(suite, "Test basic map seed", test_basic_map_seed);
    CU_add_test(suite, "Test sample data multi layer", test_sample_multi_layer);
    CU_add_test(suite, "Test map layers term", test_seed_map_layers_term);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}