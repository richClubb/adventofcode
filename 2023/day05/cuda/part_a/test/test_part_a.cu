#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "part_a.cuh"

#include "seed_map_layer.cuh"

#include <stdlib.h>

#include "config.cuh"

void test_part_a_sample(void)
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/cuda/part_a/test/part_a_sample.txt"
    };

    uint64_t result = part_a(&config);

    CU_ASSERT_EQUAL(result, 46);
}

void test_part_a_full_input(void)
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/cuda/part_a/test/input.txt"
    };

    uint64_t result = part_a(&config);

    CU_ASSERT_EQUAL(result, 525792406);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("PartATests", 0, 0);
    //CU_add_test(suite, "Test part a sample", test_part_a_sample);
    CU_add_test(suite, "Test part a full input", test_part_a_full_input);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}