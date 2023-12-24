#!/bin/env python3

from experiment import calculate_new_seeds, clean_up_seed_list
import pytest
import sys


@pytest.mark.parametrize(
    "input, expected_result",
    [
        ([(50, None), (60, None)], [(50, None), (60, None)]),  # 0
        ([(50, None), (51, None)], [(50, 51)]),
        ([(50, None), (50, None)], [(50, None)]),
        ([(50, None), (52, 53)], [(50, None), (52, 53)]),
        ([(50, 51), (53, None)], [(50, 51), (53, None)]),
        ([(50, 51), (52, None)], [(50, 52)]),  # 5
        ([(50, 51), (53, 54)], [(50, 51), (53, 54)]),
        ([(50, 51), (52, 53)], [(50, 53)]),
        ([(50, 53), (52, 53)], [(50, 53)]),
        ([(50, 54), (52, 53)], [(50, 54)]),
        ([(50, 53), (52, 54)], [(50, 54)]),  # 10
    ],
)
def test_clean_up_seed_list(input, expected_result):
    result = clean_up_seed_list(input)
    assert result == expected_result


@pytest.mark.parametrize(
    "input, expected_result",
    [
        (((50, None), [(11, 49, 2)]), ([(12, None)], None)),  # 0
        (((55, None), [(11, 49, 20)]), ([(17, None)], None)),
        (((48, None), [(11, 49, 2)]), ([(48, None)], None)),
        (((51, None), [(11, 49, 2)]), ([(51, None)], None)),
        (((30, 31), [(11, 49, 2)]), ([(30, 31)], None)),
        (((30, 48), [(11, 49, 2)]), ([(30, 48)], None)),  # 5
        (((51, 52), [(11, 49, 2)]), ([(51, 52)], None)),
        (((47, 49), [(11, 49, 2)]), ([(47, 48), (11, None)], None)),
        (((48, 49), [(11, 49, 2)]), ([(48, None), (11, None)], None)),
        (((47, 50), [(11, 49, 5)]), ([(47, 48), (11, 12)], None)),
        (((47, 52), [(11, 49, 5)]), ([(47, 48), (11, 14)], None)),  # 10
        (((47, 53), [(11, 49, 5)]), ([(47, 48), (11, 15)], None)),
        (((47, 54), [(11, 49, 5)]), ([(47, 48), (11, 15)], (54, None))),
        (((47, 55), [(11, 49, 5)]), ([(47, 48), (11, 15)], (54, 55))),
        (((49, 50), [(11, 49, 5)]), ([(11, 12)], None)),
        (((49, 52), [(11, 49, 5)]), ([(11, 14)], None)),  # 15
        (((49, 53), [(11, 49, 5)]), ([(11, 15)], None)),
        (((49, 54), [(11, 49, 5)]), ([(11, 15)], (54, None))),
        (((49, 55), [(11, 49, 5)]), ([(11, 15)], (54, 55))),
        (((50, 52), [(11, 49, 5)]), ([(12, 14)], None)),
        (((50, 53), [(11, 49, 5)]), ([(12, 15)], None)),  # 20
        (((50, 54), [(11, 49, 5)]), ([(12, 15)], (54, None))),
        (((50, 55), [(11, 49, 5)]), ([(12, 15)], (54, 55))),
        (((53, 54), [(11, 49, 5)]), ([(15, None)], (54, None))),
        (((55, 67), [(81, 45, 19)]), ([(91, 99)], (64, 67))),
        (((55, 67), [[52, 50, 48]]), ([(57, 69)], None)),
    ],
)
def test_calculate_new_seeds(input, expected_result):
    seed_range = input[0]
    mapping = input[1]

    expected_mapping = expected_result[0]
    expected_seeds = expected_result[1]

    mapping_result, remaining_seeds = calculate_new_seeds(seed_range, mapping)
    assert expected_mapping == mapping_result
    assert expected_seeds == remaining_seeds


if __name__ == "__main__":
    pytest.main(sys.argv)
