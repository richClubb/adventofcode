#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "part_b.cuh"

#include "config.cuh"

void test_part_b_sample_data_cuda()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/sample_data.txt"
    };

    uint64_t result = part_b(&config);

    CU_ASSERT_EQUAL(result, 46);
}

void test_part_b_sample_data_non_kernel()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/sample_data.txt"
    };

    uint64_t result = part_b_non_kernel(&config);

    CU_ASSERT_EQUAL(result, 46);
}

// excluding part b full test as it takes a long time

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("PartATests", 0, 0);
    CU_add_test(suite, "", test_part_b_sample_data_cuda); // still some memory leaks
    CU_add_test(suite, "", test_part_b_sample_data_non_kernel);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}