#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>

#include "part_b.h"
#include "config.h"

void test_part_b(void)
{
    CONFIG config = {
        .input_file_path = "../part_b_sample.txt",
    };

    unsigned long result = part_b(&config);

    CU_ASSERT_EQUAL(result, 46);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Part B tests", 0, 0);
    CU_add_test(suite, "Test part b", test_part_b);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}