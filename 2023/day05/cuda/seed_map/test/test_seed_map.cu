#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdint.h>

#include "seed.cuh"
#include "seed_map.cuh"

void test_seed_map_from_string()
{
    {
        SEED_MAP *seed_map = seed_map_from_string("1 2 3\n");
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);
        CU_ASSERT_EQUAL(seed_map->source_start, 2);
        CU_ASSERT_EQUAL(seed_map->source_end,   5);
        CU_ASSERT_EQUAL(seed_map->target_start, 1);
        CU_ASSERT_EQUAL(seed_map->target_end,   4);
        CU_ASSERT_EQUAL(seed_map->size,         3);
        free(seed_map);
    }

    {
        SEED_MAP *seed_map = seed_map_from_string("4 5 6\n");
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);
        CU_ASSERT_EQUAL(seed_map->source_start, 5);
        CU_ASSERT_EQUAL(seed_map->source_end,   11);
        CU_ASSERT_EQUAL(seed_map->target_start, 4);
        CU_ASSERT_EQUAL(seed_map->target_end,   10);
        CU_ASSERT_EQUAL(seed_map->size,         6);
        free(seed_map);
    }

    {
        SEED_MAP *seed_map = seed_map_from_string("7 8 9\n");
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);
        CU_ASSERT_EQUAL(seed_map->source_start, 8);
        CU_ASSERT_EQUAL(seed_map->source_end,   17);
        CU_ASSERT_EQUAL(seed_map->target_start, 7);
        CU_ASSERT_EQUAL(seed_map->target_end,   16);
        CU_ASSERT_EQUAL(seed_map->size,         9);
        free(seed_map);
    }

    {
        SEED_MAP *seed_map = seed_map_from_string("1633669237 1273301814 72865265\n");
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);
        CU_ASSERT_EQUAL(seed_map->source_start, 1273301814);
        CU_ASSERT_EQUAL(seed_map->source_end,   1346167079);
        CU_ASSERT_EQUAL(seed_map->target_start, 1633669237);
        CU_ASSERT_EQUAL(seed_map->target_end,   1706534502);
        CU_ASSERT_EQUAL(seed_map->size,         72865265);
        free(seed_map);
    }
}

void test_seed_map_from_string_errors()
{
    {
        SEED_MAP *seed_map = seed_map_from_string("1 2");
        CU_ASSERT_PTR_NULL(seed_map);
    }

    {
        SEED_MAP *seed_map = seed_map_from_string("4 5 6 6\n");
        CU_ASSERT_PTR_NULL(seed_map);
    }
}

void test_seed_map_map_seed()
{

    SEED_MAP seed_map{
        .source_start = 2,
        .source_end = 12,
        .target_start = 22,
        .target_end = 32,
        .size = 10
    };

    {
        SEED value = 1;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, false);
        CU_ASSERT_EQUAL(value, 1);
    }

    {
        SEED value = 2;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 22);
    }

    {
        SEED value = 3;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 23);
    }

    {
        SEED value = 4;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 24);
    }

    {
        SEED value = 5;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 25);
    }

    {
        SEED value = 6;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 26);
    }

    {
        SEED value = 7;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 27);
    }

    {
        SEED value = 8;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 28);
    }

    {
        SEED value = 9;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 29);
    }

    {
        SEED value = 10;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 30);
    }

    {
        SEED value = 11;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, true);
        CU_ASSERT_EQUAL(value, 31);
    }

    {
        SEED value = 12;
        bool result = seed_map_map_seed(&seed_map, &value);
        CU_ASSERT_EQUAL(result, false);
        CU_ASSERT_EQUAL(value, 12);
    }
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Seed Tests", 0, 0);
    CU_add_test(suite, "", test_seed_map_from_string);
    CU_add_test(suite, "", test_seed_map_from_string_errors);
    CU_add_test(suite, "", test_seed_map_map_seed);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}