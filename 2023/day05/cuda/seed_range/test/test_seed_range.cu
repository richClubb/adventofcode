#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdlib.h>
#include <stdint.h>

#include "seed_range.cuh"

void test_get_seed_ranges(void)
{
    char *input_line = "seeds: 10 15\n";

    unsigned int num_seed_ranges = 0;
    SEED_RANGE **seed_ranges = get_seed_ranges(input_line, &num_seed_ranges);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

    CU_ASSERT_EQUAL(num_seed_ranges, 1);
    CU_ASSERT_EQUAL(seed_ranges[0]->start, 10);
    CU_ASSERT_EQUAL(seed_ranges[0]->size,  15);

    seed_ranges_term(seed_ranges, num_seed_ranges);
}

void test_get_seed_ranges_2(void)
{
    char *input_line = "seeds: 10 15 30 35\n";

    unsigned int num_seed_ranges = 0;
    SEED_RANGE **seed_ranges = get_seed_ranges(input_line, &num_seed_ranges);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

    CU_ASSERT_EQUAL(num_seed_ranges, 2);

    CU_ASSERT_EQUAL(seed_ranges[0]->start, 10);
    CU_ASSERT_EQUAL(seed_ranges[0]->size,  15);

    CU_ASSERT_EQUAL(seed_ranges[1]->start, 30);
    CU_ASSERT_EQUAL(seed_ranges[1]->size,  35);

    seed_ranges_term(seed_ranges, num_seed_ranges);
}

void test_get_seed_ranges_3(void)
{
    char *input_line = "seeds: 28965817 302170009 1752849261 48290258 804904201 243492043 2150339939 385349830 1267802202 350474859 2566296746 17565716 3543571814 291402104 447111316 279196488 3227221259 47952959 1828835733 9607836\n";

    unsigned int num_seed_ranges = 0;
    SEED_RANGE **seed_ranges = get_seed_ranges(input_line, &num_seed_ranges);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

    CU_ASSERT_EQUAL(num_seed_ranges, 10);

    CU_ASSERT_EQUAL(seed_ranges[0]->start, 28965817);
    CU_ASSERT_EQUAL(seed_ranges[0]->size,  302170009);

    CU_ASSERT_EQUAL(seed_ranges[1]->start, 1752849261);
    CU_ASSERT_EQUAL(seed_ranges[1]->size,  48290258);

    CU_ASSERT_EQUAL(seed_ranges[2]->start, 804904201);
    CU_ASSERT_EQUAL(seed_ranges[2]->size,  243492043);

    CU_ASSERT_EQUAL(seed_ranges[3]->start, 2150339939);
    CU_ASSERT_EQUAL(seed_ranges[3]->size,  385349830);

    CU_ASSERT_EQUAL(seed_ranges[4]->start, 1267802202);
    CU_ASSERT_EQUAL(seed_ranges[4]->size,  350474859);

    CU_ASSERT_EQUAL(seed_ranges[5]->start, 2566296746);
    CU_ASSERT_EQUAL(seed_ranges[5]->size,  17565716);

    CU_ASSERT_EQUAL(seed_ranges[6]->start, 3543571814);
    CU_ASSERT_EQUAL(seed_ranges[6]->size,  291402104);

    CU_ASSERT_EQUAL(seed_ranges[7]->start, 447111316);
    CU_ASSERT_EQUAL(seed_ranges[7]->size,  279196488);

    CU_ASSERT_EQUAL(seed_ranges[8]->start, 3227221259);
    CU_ASSERT_EQUAL(seed_ranges[8]->size,  47952959);

    CU_ASSERT_EQUAL(seed_ranges[9]->start, 1828835733);
    CU_ASSERT_EQUAL(seed_ranges[9]->size,  9607836);

    seed_ranges_term(seed_ranges, num_seed_ranges);
}

void test_split_seed_ranges(void)
{
    char *input_line = "seeds: 28965817 302170009 1752849261 48290258\n";

    unsigned int num_seed_ranges = 0;
    SEED_RANGE **seed_ranges = get_seed_ranges(input_line, &num_seed_ranges);

    CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

    CU_ASSERT_EQUAL(num_seed_ranges, 2);

    SEED_RANGE **new_ranges = seed_ranges_split_by_size(seed_ranges, num_seed_ranges, 10);

    CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);

    seed_ranges_term(seed_ranges, num_seed_ranges);
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Seed Range tests", 0, 0);
    // CU_add_test(suite, "Test get seed ranges", test_get_seed_ranges);
    // CU_add_test(suite, "Test get seed ranges 2", test_get_seed_ranges_2);
    // CU_add_test(suite, "Test get seed ranges 3", test_get_seed_ranges_3);
    CU_add_test(suite, "Test split_seed_ranges", test_split_seed_ranges);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}