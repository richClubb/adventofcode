#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

void basic_assert(void)
{
    CU_ASSERT(CU_TRUE);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("ConfigTests", 0, 0);
    CU_add_test(suite, "Basic_assert", basic_assert);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}