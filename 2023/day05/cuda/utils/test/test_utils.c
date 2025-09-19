#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>

#include "utils.h"

#define TEST

void basic_assert(void)
{
    CU_ASSERT_TRUE(CU_TRUE);
}

void test_extract_numbers(void)
{
    char numbers_string[] = "1 10 6\n";

    unsigned int num_numbers = 0;

    unsigned long *numbers = extract_number_list(numbers_string, &num_numbers);

    CU_ASSERT_PTR_NOT_NULL_FATAL(numbers);
    CU_ASSERT_EQUAL(num_numbers, 3);
    CU_ASSERT_EQUAL(numbers[0], 1);
    CU_ASSERT_EQUAL(numbers[1], 10);
    CU_ASSERT_EQUAL(numbers[2], 6);

    free(numbers);
}

void test_get_seeds(void)
{
    char *seeds_line = "seeds: 1 2 3 10 100\n";

    unsigned int num_seeds = 0;
    unsigned long *seeds = get_seeds(seeds_line, &num_seeds);

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

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Utils tests", 0, 0);
    CU_add_test(suite, "Basic assert", basic_assert);
    CU_add_test(suite, "Basic extract numbers", test_extract_numbers);
    CU_add_test(suite, "Basic get seeds", test_get_seeds);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}