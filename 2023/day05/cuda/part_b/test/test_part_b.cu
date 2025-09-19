#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "part_b.cuh"

#include "seed_map_layer.cuh"

#include <stdlib.h>

#include "config.cuh"

void test_part_b_sample(void)
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/cuda/part_b/test/part_a_sample.txt"
    };

    uint64_t result = part_b(&config);

    CU_ASSERT_EQUAL(result, 46);
}

void test_part_b_full(void)
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/cuda/part_b/test/input.txt"
    };

    uint64_t result = part_b(&config);

    CU_ASSERT_EQUAL(result, 35);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("PartATests", 0, 0);
    CU_add_test(suite, "Test part a sample", test_part_b_sample);
    //CU_add_test(suite, "Test part a full", test_part_b_full);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}