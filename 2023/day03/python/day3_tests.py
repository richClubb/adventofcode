#!/bin/env python3
import pytest
import sys
from day3 import part_a


def test_part_a():
    assert part_a("../part_a_sample.txt") == 4361


if __name__ == "__main__":
    pytest.main(sys.argv)
