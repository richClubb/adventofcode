#!/bin/env python3

from day5 import clean_up_seed_list
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
def test_calculate_new_seeds(input, expected_result):
    result = clean_up_seed_list(input)
    assert result == expected_result


if __name__ == "__main__":
    pytest.main(sys.argv)
