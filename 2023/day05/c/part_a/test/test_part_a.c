#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>

#include "part_a.h"
#include "test_header_part_a.h"

#define TEST

void basic_assert(void)
{
    CU_ASSERT_TRUE(CU_TRUE);
}

void test_get_seeds(void)
{
    char *seeds_line = "seeds: 1 2 3 10 100\n";

    unsigned int num_seeds = 0;
    long *seeds = get_seeds(seeds_line, &num_seeds);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seeds);

    CU_ASSERT_EQUAL(num_seeds, 5);

    CU_ASSERT_EQUAL(seeds[0], 1);
    CU_ASSERT_EQUAL(seeds[1], 2);
    CU_ASSERT_EQUAL(seeds[2], 3);
    CU_ASSERT_EQUAL(seeds[3], 10);
    CU_ASSERT_EQUAL(seeds[4], 100);

    // free seeds which was allocated in get_seeds
    free(seeds);
}

void test_get_seed_map(void)
{
    char *seed_map_line = "1 10 5\n";

    SEED_MAP *seed_map = get_seed_map(seed_map_line);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_map);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Part A tests", 0, 0);
    CU_add_test(suite, "Basic assert", basic_assert);
    CU_add_test(suite, "Test get seeds", test_get_seeds);
    CU_add_test(suite, "Test get seeds", test_get_seed_map);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}