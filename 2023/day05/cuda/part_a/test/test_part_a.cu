#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "part_a.cuh"

#include "config.cuh"

void test_part_a_sample_data_cuda()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/sample_data.txt"
    };

    uint64_t result = part_a(&config);

    CU_ASSERT_EQUAL(result, 35);
}

void test_part_a_sample_data_non_cuda()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/sample_data.txt"
    };

    uint64_t result = part_a_non_kernel(&config);

    CU_ASSERT_EQUAL(result, 35);
}

void test_part_a_full_data_cuda()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/full_data.txt"
    };

    uint64_t result = part_a(&config);

    CU_ASSERT_EQUAL(result, 525792406);
}

void test_part_a_full_data_non_cuda()
{
    CONFIG config = {
        .input_file_path = "/workspaces/adventofcode/2023/day05/full_data.txt"
    };

    uint64_t result = part_a_non_kernel(&config);

    CU_ASSERT_EQUAL(result, 525792406);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("PartATests", 0, 0);
    CU_add_test(suite, "Test part a sample cuda", test_part_a_sample_data_cuda);
    CU_add_test(suite, "Test part a sample non cuda", test_part_a_sample_data_non_cuda);
    CU_add_test(suite, "Test part a full input cuda", test_part_a_full_data_cuda);
    CU_add_test(suite, "Test part a full input non cuda", test_part_a_full_data_non_cuda);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}