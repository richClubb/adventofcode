#!/bin/env python3

from experiment1 import calculate_new_seeds
import pytest
import sys


@pytest.mark.parametrize(
    "input, expected_result",
    [
        (((1, 10), [(11, 1, 10)]), ([(11, 20)], None)),
        (((50, 60), [(11, 49, 3)]), ([(12, 13)], [(52, 60)])),
        (((50, 60), [(11, 49, 2)]), ([(12)], [(51, 60)])),
        (((50, 51), [(11, 49, 2)]), ([(12)], [(51)])),
        (((50), [(11, 49, 2)]), ([(12)], None)),
        (((55), [(11, 49, 20)]), ([(17)], None)),
        (((50, 60), [(11, 55, 3)]), ([(50, 54), (11, 13)], [(58, 60)])),
        (((50, 60), [(11, 55, 5)]), ([(50, 54), (11, 15)], [60])),
        (((54, 60), [(11, 55, 3)]), ([54, (11, 13)], [(58, 60)])),
        (((54, 60), [(11, 55, 5)]), ([54, (11, 15)], [60])),
        (((50, 55), [(11, 1, 10)]), ([(50, 55)], None)),
        (((50, 55), [(11, 70, 10)]), ([(50, 55)], None)),
        (((50, 55), [(11, 1, 10), (11, 70, 10)]), ([(50, 55)], None)),
        (((50, 55), [(11, 53, 3)]), ([(50, 52), (11, 13)], None)),
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
