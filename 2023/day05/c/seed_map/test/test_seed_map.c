#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map.h"

void basic_map_1(void)
{
    SEED_MAP test_map = {
        .source = 1,
        .target = 10,
        .size = 5
    };

    long value = 0;
    CU_ASSERT_FALSE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 0);

    value = 1;
    CU_ASSERT_TRUE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 10);

    value = 2;
    CU_ASSERT_TRUE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 11);

    value = 3;
    CU_ASSERT_TRUE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 12);

    value = 4;
    CU_ASSERT_TRUE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 13);

    value = 5;
    CU_ASSERT_TRUE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 14);

    value = 6;
    CU_ASSERT_FALSE(seed_map_map_seed(&test_map, &value));
    CU_ASSERT_EQUAL(value, 6);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapTests", 0, 0);
    CU_add_test(suite, "BasicMap_1", basic_map_1);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}