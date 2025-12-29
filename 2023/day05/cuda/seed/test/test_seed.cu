#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdint.h>

#include "seed.cuh"

void test_get_seeds()
{
    {
        uint64_t num_seeds = 0;
        SEED *seeds_error = seed_get_seeds_from_line("seeds 1\n", &num_seeds);
        CU_ASSERT_EQUAL(num_seeds, 0);
        CU_ASSERT_PTR_NULL_FATAL(seeds_error);
        free(seeds_error);
    }
    
    {
        uint64_t num_seeds = 0;
        SEED *seeds = seed_get_seeds_from_line("seeds: 1\n", &num_seeds);
        CU_ASSERT_EQUAL(num_seeds, 1);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seeds);
        CU_ASSERT_EQUAL(seeds[0], 1);
        free(seeds);
    }
    
    {
        uint64_t num_seeds = 0;
        SEED *seeds = seed_get_seeds_from_line("seeds: 1 2\n", &num_seeds);
        CU_ASSERT_EQUAL_FATAL(num_seeds, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seeds);
        CU_ASSERT_EQUAL(seeds[0], 1);
        CU_ASSERT_EQUAL(seeds[1], 2);
        free(seeds);
    }
    
    {
        uint64_t num_seeds = 0;
        SEED *seeds = seed_get_seeds_from_line("seeds: 1 2 3\n", &num_seeds);
        CU_ASSERT_EQUAL_FATAL(num_seeds, 3);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seeds);
        CU_ASSERT_EQUAL(seeds[0], 1);
        CU_ASSERT_EQUAL(seeds[1], 2);
        CU_ASSERT_EQUAL(seeds[2], 3);
        free(seeds);
    }

    {
        uint64_t num_seeds = 0;
        SEED *seeds = seed_get_seeds_from_line("seeds: 1 2 3 5000000000\n", &num_seeds);
        CU_ASSERT_EQUAL_FATAL(num_seeds, 4);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seeds);
        CU_ASSERT_EQUAL(seeds[0], 1);
        CU_ASSERT_EQUAL(seeds[1], 2);
        CU_ASSERT_EQUAL(seeds[2], 3);
        CU_ASSERT_EQUAL(seeds[3], 5000000000);
        free(seeds);
    }
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Seed Tests", 0, 0);
    CU_add_test(suite, "", test_get_seeds);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}