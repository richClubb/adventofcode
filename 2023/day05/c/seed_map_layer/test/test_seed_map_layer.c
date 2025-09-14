#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map_layer.h"

void basic_assert(void)
{
    CU_ASSERT(CU_TRUE);
}

void test_init(void)
{
    SEED_MAP_LAYER seed_map_layer;

    seed_map_layer_init(&seed_map_layer);

    CU_ASSERT_EQUAL_FATAL(seed_map_layer.seed_map_count, 0);
    CU_ASSERT_EQUAL_FATAL(seed_map_layer.seed_maps, NULL);
}

void test_add_1_map(void)
{
    SEED_MAP_LAYER seed_map_layer;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP seed_map_1 = {
        .source = 1,
        .target = 10,
        .size = 5
    };

    seed_map_layer_add_seed_map(&seed_map_layer, &seed_map_1);

    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->source, 1);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->target, 10);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->size, 5);
}

void test_add_2_maps(void)
{
    SEED_MAP_LAYER seed_map_layer;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP seed_map_1 = {
        .source = 1,
        .target = 10,
        .size = 5
    };

    SEED_MAP seed_map_2 = {
        .source = 20,
        .target = 7,
        .size = 2
    };

    seed_map_layer_add_seed_map(&seed_map_layer, &seed_map_1);
    seed_map_layer_add_seed_map(&seed_map_layer, &seed_map_2);

    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->source, 1);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->target, 10);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[0]->size, 5);

    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[1]->source, 20);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[1]->target, 7);
    CU_ASSERT_EQUAL(seed_map_layer.seed_maps[1]->size, 2);
}

void test_basic_map_seed(void)
{
    SEED_MAP_LAYER seed_map_layer;

    seed_map_layer_init(&seed_map_layer);

    SEED_MAP seed_map_1 = {
        .source = 1,
        .target = 10,
        .size = 5
    };

    seed_map_layer_add_seed_map(&seed_map_layer, &seed_map_1);

    long value = 0;
    CU_ASSERT_FALSE(seed_map_layer_map_seed(&seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 0);

    value = 1;
    CU_ASSERT_TRUE(seed_map_layer_map_seed(&seed_map_layer, &value));
    CU_ASSERT_EQUAL(value, 10);
}

// not aure cunit can catch this
void test_null_seed_map_layer()
{
    long value = 0;
    seed_map_layer_map_seed(NULL, &value);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapLayerTests", 0, 0);
    CU_add_test(suite, "Basic_assert", basic_assert);
    CU_add_test(suite, "Test init map layer", test_init);
    CU_add_test(suite, "Test add 1 map to layer", test_add_1_map);
    CU_add_test(suite, "Test add 2 maps to layer", test_add_2_maps);
    CU_add_test(suite, "Test basic map seed", test_basic_map_seed);
    CU_add_test(suite, "Test assert null seed map layer", test_basic_map_seed);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}