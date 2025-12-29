#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>

#include "part_a.h"

#include "config.h"

void test_part_a(void)
{
    CONFIG config = {
        .input_file_path = "../part_a_sample.txt",
    };

    unsigned long result = part_a(&config);

    CU_ASSERT_EQUAL(result, 35);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Part A tests", 0, 0);
    
    CU_add_test(suite, "Test part a", test_part_a);

    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}