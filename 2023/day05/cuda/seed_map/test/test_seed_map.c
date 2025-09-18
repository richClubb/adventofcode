#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include "seed_map_layer.h"

#include <stdlib.h>

void test_init(void)
{
    
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("SeedMapLayerTests", 0, 0);
    CU_add_test(suite, "Test init map layer", test_init);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}