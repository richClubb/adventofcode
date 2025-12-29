#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>

#include "seed_map.h"

void basic_map_1(void)
{
    SEED_MAP *test_map = NULL;

    seed_map_init(&test_map);
    test_map->source = 1;
    test_map->target = 10;
    test_map->size = 5;
    
    unsigned long value = 0;
    CU_ASSERT_FALSE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 0);

    value = 1;
    CU_ASSERT_TRUE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 10);

    value = 2;
    CU_ASSERT_TRUE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 11);

    value = 3;
    CU_ASSERT_TRUE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 12);

    value = 4;
    CU_ASSERT_TRUE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 13);

    value = 5;
    CU_ASSERT_TRUE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 14);

    value = 6;
    CU_ASSERT_FALSE(seed_map_map_seed(test_map, &value));
    CU_ASSERT_EQUAL(value, 6);

    seed_map_term(test_map);
}

void test_get_seed_map(void)
{
    char *seed_map_line = "1 10 5\n";

    SEED_MAP *seed_map = get_seed_map(seed_map_line);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);

    CU_ASSERT_EQUAL(seed_map->source, 10);
    CU_ASSERT_EQUAL(seed_map->target, 1);
    CU_ASSERT_EQUAL(seed_map->size,   5);

    seed_map_term(seed_map);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapTests", 0, 0);
    CU_add_test(suite, "BasicMap_1", basic_map_1);
    CU_add_test(suite, "Test get seed map", test_get_seed_map);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}