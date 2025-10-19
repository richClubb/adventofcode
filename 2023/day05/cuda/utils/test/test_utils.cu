#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdint.h>

#include "utils.cuh"

void test_extract_numbers()
{
    {
        uint64_t length = 0;
        uint64_t *numbers = extract_number_list("1\n", &length);
    
        CU_ASSERT_EQUAL(length, 1);
        CU_ASSERT_PTR_NOT_NULL_FATAL(numbers);
        CU_ASSERT_EQUAL(numbers[0], 1);
        free(numbers);
    }

    {
        uint64_t length = 0;
        uint64_t *numbers = extract_number_list("1 2\n", &length);
    
        CU_ASSERT_EQUAL(length, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(numbers);
        CU_ASSERT_EQUAL(numbers[0], 1);
        CU_ASSERT_EQUAL(numbers[1], 2);
        free(numbers);
    }

    {
        uint64_t length = 0;
        uint64_t *numbers = extract_number_list("1 2 3 5 8 1000000\n", &length);
    
        CU_ASSERT_EQUAL(length, 6);
        CU_ASSERT_PTR_NOT_NULL_FATAL(numbers);
        CU_ASSERT_EQUAL(numbers[0], 1);
        CU_ASSERT_EQUAL(numbers[1], 2);
        CU_ASSERT_EQUAL(numbers[2], 3);
        CU_ASSERT_EQUAL(numbers[3], 5);
        CU_ASSERT_EQUAL(numbers[4], 8);
        CU_ASSERT_EQUAL(numbers[5], 1000000);
        free(numbers);
    }
}

void test_extract_numbers_errors()
{
    {
        uint64_t length = 0;
        uint64_t *numbers = extract_number_list("1 a\n", &length);
    
        CU_ASSERT_EQUAL(length, 0);
        CU_ASSERT_PTR_NULL_FATAL(numbers);
        free(numbers);
    }
}


int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Utils Tests", 0, 0);
    CU_add_test(suite, "", test_extract_numbers);
    //CU_add_test(suite, "", test_extract_numbers_errors);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}