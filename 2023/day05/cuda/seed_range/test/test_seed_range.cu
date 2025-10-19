#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#include <stdint.h>

#include "seed_range.cuh"

void test_get_seed_ranges()
{
    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 1 2 3 4", &num_seed_ranges);

        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);
        CU_ASSERT_EQUAL(seed_ranges[0].start, 1);
        CU_ASSERT_EQUAL(seed_ranges[0].size, 2);
        CU_ASSERT_EQUAL(seed_ranges[1].start, 3);
        CU_ASSERT_EQUAL(seed_ranges[1].size, 4);
    }

    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 1 2 3 4 5000000000 1000000000", &num_seed_ranges);

        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 3);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);
        CU_ASSERT_EQUAL(seed_ranges[0].start, 1);
        CU_ASSERT_EQUAL(seed_ranges[0].size, 2);
        CU_ASSERT_EQUAL(seed_ranges[1].start, 3);
        CU_ASSERT_EQUAL(seed_ranges[1].size, 4);
        CU_ASSERT_EQUAL(seed_ranges[2].start, 5000000000);
        CU_ASSERT_EQUAL(seed_ranges[2].size, 1000000000);
    }
}

void test_sort_by_size()
{
    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 1 2 5 6 3 3", &num_seed_ranges);
        
        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 3);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

        sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);

        CU_ASSERT_EQUAL(seed_ranges[0].start, 1);
        CU_ASSERT_EQUAL(seed_ranges[0].size,  2);
        CU_ASSERT_EQUAL(seed_ranges[1].start, 3);
        CU_ASSERT_EQUAL(seed_ranges[1].size,  3);
        CU_ASSERT_EQUAL(seed_ranges[2].start, 5);
        CU_ASSERT_EQUAL(seed_ranges[2].size,  6);
    }
}

void test_sort_by_start()
{
    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 1 2 5 6 3 3", &num_seed_ranges);
        
        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 3);
        CU_ASSERT_PTR_NOT_NULL_FATAL(seed_ranges);

        sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);

        CU_ASSERT_EQUAL(seed_ranges[0].start, 1);
        CU_ASSERT_EQUAL(seed_ranges[0].size,  2);
        CU_ASSERT_EQUAL(seed_ranges[1].start, 3);
        CU_ASSERT_EQUAL(seed_ranges[1].size,  3);
        CU_ASSERT_EQUAL(seed_ranges[2].start, 5);
        CU_ASSERT_EQUAL(seed_ranges[2].size,  6);
    }
}

void test_ideal_size()
{
    CU_ASSERT_EQUAL(ideal_size(20, 50), 20);
    CU_ASSERT_EQUAL(ideal_size(20, 20), 20);
    CU_ASSERT_EQUAL(ideal_size(19, 50), 19);
    CU_ASSERT_EQUAL(ideal_size(19, 19), 19);
    CU_ASSERT_EQUAL(ideal_size(20, 1), 1);
    CU_ASSERT_EQUAL(ideal_size(19, 1), 1);

    CU_ASSERT_EQUAL(ideal_size(20, 19), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 18), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 17), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 16), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 15), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 14), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 13), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 12), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 11), 10);
    CU_ASSERT_EQUAL(ideal_size(20, 10), 10);
    CU_ASSERT_EQUAL(ideal_size(20,  9),  7);
    CU_ASSERT_EQUAL(ideal_size(20,  8),  7);
    CU_ASSERT_EQUAL(ideal_size(20,  7),  7);
    CU_ASSERT_EQUAL(ideal_size(20,  6),  5);
    CU_ASSERT_EQUAL(ideal_size(20,  5),  5);
    CU_ASSERT_EQUAL(ideal_size(20,  4),  4);
    CU_ASSERT_EQUAL(ideal_size(20,  3),  3);
    CU_ASSERT_EQUAL(ideal_size(20,  2),  2);

    CU_ASSERT_EQUAL(ideal_size(19, 18), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 17), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 16), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 15), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 14), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 13), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 12), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 11), 10);
    CU_ASSERT_EQUAL(ideal_size(19, 10), 10);
    CU_ASSERT_EQUAL(ideal_size(19,  9),  7);
    CU_ASSERT_EQUAL(ideal_size(19,  8),  7);
    CU_ASSERT_EQUAL(ideal_size(19,  7),  7);
    CU_ASSERT_EQUAL(ideal_size(19,  6),  5);
    CU_ASSERT_EQUAL(ideal_size(19,  5),  5);
    CU_ASSERT_EQUAL(ideal_size(19,  4),  4);
    CU_ASSERT_EQUAL(ideal_size(19,  3),  3);
    CU_ASSERT_EQUAL(ideal_size(19,  2),  2);
}

void test_split_seed_range_by_size()
{
    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 10, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 1);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 10);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 9, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 5);
        CU_ASSERT_EQUAL(new_ranges[1].start, 6);
        CU_ASSERT_EQUAL(new_ranges[1].size, 5);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 8, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 5);
        CU_ASSERT_EQUAL(new_ranges[1].start, 6);
        CU_ASSERT_EQUAL(new_ranges[1].size, 5);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 7, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 5);
        CU_ASSERT_EQUAL(new_ranges[1].start, 6);
        CU_ASSERT_EQUAL(new_ranges[1].size, 5);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 6, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 5);
        CU_ASSERT_EQUAL(new_ranges[1].start, 6);
        CU_ASSERT_EQUAL(new_ranges[1].size, 5);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 5, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 2);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 5);
        CU_ASSERT_EQUAL(new_ranges[1].start, 6);
        CU_ASSERT_EQUAL(new_ranges[1].size, 5);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 4, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 3);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 4);
        CU_ASSERT_EQUAL(new_ranges[1].start, 5);
        CU_ASSERT_EQUAL(new_ranges[1].size, 3);
        CU_ASSERT_EQUAL(new_ranges[2].start, 8);
        CU_ASSERT_EQUAL(new_ranges[2].size, 3);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 3, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 4);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 3);
        CU_ASSERT_EQUAL(new_ranges[1].start, 4);
        CU_ASSERT_EQUAL(new_ranges[1].size, 3);
        CU_ASSERT_EQUAL(new_ranges[2].start, 7);
        CU_ASSERT_EQUAL(new_ranges[2].size, 2);
        CU_ASSERT_EQUAL(new_ranges[3].start, 9);
        CU_ASSERT_EQUAL(new_ranges[3].size, 2);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 2, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 5);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 2);
        CU_ASSERT_EQUAL(new_ranges[1].start, 3);
        CU_ASSERT_EQUAL(new_ranges[1].size, 2);
        CU_ASSERT_EQUAL(new_ranges[2].start, 5);
        CU_ASSERT_EQUAL(new_ranges[2].size, 2);
        CU_ASSERT_EQUAL(new_ranges[3].start, 7);
        CU_ASSERT_EQUAL(new_ranges[3].size, 2);
        CU_ASSERT_EQUAL(new_ranges[4].start, 9);
        CU_ASSERT_EQUAL(new_ranges[4].size, 2);
        free(new_ranges);
    }

    {
        SEED_RANGE range = {
            .start = 1,
            .size = 10
        };

        uint64_t range_count = 0;
        SEED_RANGE *new_ranges = split_seed_range_by_size(&range, 1, &range_count);
        CU_ASSERT_EQUAL_FATAL(range_count, 10);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_ranges);
        CU_ASSERT_EQUAL(new_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_ranges[0].size, 1);
        CU_ASSERT_EQUAL(new_ranges[1].start, 2);
        CU_ASSERT_EQUAL(new_ranges[1].size, 1);
        CU_ASSERT_EQUAL(new_ranges[2].start, 3);
        CU_ASSERT_EQUAL(new_ranges[2].size, 1);
        CU_ASSERT_EQUAL(new_ranges[3].start, 4);
        CU_ASSERT_EQUAL(new_ranges[3].size, 1);
        CU_ASSERT_EQUAL(new_ranges[4].start, 5);
        CU_ASSERT_EQUAL(new_ranges[4].size, 1);
        CU_ASSERT_EQUAL(new_ranges[5].start, 6);
        CU_ASSERT_EQUAL(new_ranges[5].size, 1);
        CU_ASSERT_EQUAL(new_ranges[6].start, 7);
        CU_ASSERT_EQUAL(new_ranges[6].size, 1);
        CU_ASSERT_EQUAL(new_ranges[7].start, 8);
        CU_ASSERT_EQUAL(new_ranges[7].size, 1);
        CU_ASSERT_EQUAL(new_ranges[8].start, 9);
        CU_ASSERT_EQUAL(new_ranges[8].size, 1);
        CU_ASSERT_EQUAL(new_ranges[9].start, 10);
        CU_ASSERT_EQUAL(new_ranges[9].size, 1);
        free(new_ranges);
    }
}

void test_split_seed_ranges_by_number()
{
    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 1 20 31 30", &num_seed_ranges);

        SEED_RANGE *new_seed_ranges = split_seed_ranges_by_number(seed_ranges, &num_seed_ranges, 5);
        free(seed_ranges);

        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 5);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_seed_ranges);

        CU_ASSERT_EQUAL(new_seed_ranges[0].start, 1);
        CU_ASSERT_EQUAL(new_seed_ranges[0].size, 10);
        CU_ASSERT_EQUAL(new_seed_ranges[1].start,11);
        CU_ASSERT_EQUAL(new_seed_ranges[1].size, 10);
        CU_ASSERT_EQUAL(new_seed_ranges[2].start,31);
        CU_ASSERT_EQUAL(new_seed_ranges[2].size, 10);
        CU_ASSERT_EQUAL(new_seed_ranges[3].start,41);
        CU_ASSERT_EQUAL(new_seed_ranges[3].size, 10);
        CU_ASSERT_EQUAL(new_seed_ranges[4].start,51);
        CU_ASSERT_EQUAL(new_seed_ranges[4].size, 10);
        free(new_seed_ranges);
    }

    {
        uint64_t num_seed_ranges = 0;
        SEED_RANGE *seed_ranges = get_seed_ranges_from_line("seeds: 79 14 55 13", &num_seed_ranges);
        
        sort_seed_ranges_by_size(seed_ranges, num_seed_ranges);

        SEED_RANGE *new_seed_ranges = split_seed_ranges_by_number(seed_ranges, &num_seed_ranges, 3000);
        free(seed_ranges);

        CU_ASSERT_EQUAL_FATAL(num_seed_ranges, 27);
        CU_ASSERT_PTR_NOT_NULL_FATAL(new_seed_ranges);
        
        CU_ASSERT_EQUAL(new_seed_ranges[0].start, 55);
        CU_ASSERT_EQUAL(new_seed_ranges[0].size,   1);

        CU_ASSERT_EQUAL(new_seed_ranges[26].start, 92);
        CU_ASSERT_EQUAL(new_seed_ranges[26].size,   1);
        free(new_seed_ranges);
    }
}

int main()
{
    CU_initialize_registry();
    CU_pSuite suite = CU_add_suite("Seed Tests", 0, 0);
    CU_add_test(suite, "", test_get_seed_ranges);
    CU_add_test(suite, "", test_sort_by_size);
    CU_add_test(suite, "", test_sort_by_start);

    CU_add_test(suite, "", test_ideal_size);

    CU_add_test(suite, "", test_split_seed_range_by_size);
    
    CU_add_test(suite, "", test_split_seed_ranges_by_number);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return 0;
}