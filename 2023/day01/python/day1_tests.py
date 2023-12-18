#!/bin/env python3
import pytest
import sys

from day1 import (
    is_number,
    first_last_num,
    convert_words_to_numbers,
    process_file_a,
    process_file_b,
)


@pytest.mark.parametrize(
    "value, expected_result",
    [
        ("a", False),
        ("1", True),
        ("2", True),
        ("3", True),
        ("4", True),
        ("5", True),
        ("6", True),
        ("7", True),
        ("8", True),
        ("9", True),
        ("<", False),
        (">", False),
        ("#", False),
        (" ", False),
        ("!", False),
        ("F", False),
    ],
)
def test_is_number(value, expected_result):
    assert is_number(value) == expected_result


@pytest.mark.parametrize(
    "input_list, expected_result",
    [([1, 3], 13), ([1], 11), ([1, 2, 3], 13)],
)
def test_first_last_num(input_list, expected_result):
    assert first_last_num(input_list) == expected_result


@pytest.mark.parametrize(
    "input_string, expected_result",
    [
        ("one", "1ne"),
        ("1", "1"),
        ("one1", "1ne1"),
        ("nineninenine", "9ine9ine9ine"),
        ("one1zzztwo8", "1ne1zzz2wo8"),
        ("nine57", "9ine57"),
        ("7rdtplhbvddvlkonefqsttj", "7rdtplhbvddvlk1nefqsttj"),
        ("four6mssqzhgxt", "4our6mssqzhgxt"),
        ("gphnqxhlhzzftghk767", "gphnqxhlhzzftghk767"),
        ("mbcbpjcsnt4six", "mbcbpjcsnt46ix"),
        ("one9xmhvzklmzffive1kcsixmnsbm2", "1ne9xmhvzklmzf5ive1kc6ixmnsbm2"),
        ("1dgschj", "1dgschj"),
        ("nine8foursnczninednds", "9ine84oursncz9inednds"),
        (
            "9sevensixrsrgmclkvthkgtxqtwovtlxfrdnllxvsghslh",
            "97even6ixrsrgmclkvthkgtxq2wovtlxfrdnllxvsghslh",
        ),
        ("seven443six8three31", "7even4436ix83hree31"),
        (
            "pppmfmnfourtworxrqrfhbgx8vvxgrjzhvqmztltwo",
            "pppmfmn4our2worxrqrfhbgx8vvxgrjzhvqmztl2wo",
        ),
        ("56oneninethreevv4chvlfljbrthree", "561ne9ine3hreevv4chvlfljbr3hree"),
        ("9sixseven9zspvdsqxzf", "96ix7even9zspvdsqxzf"),
        ("4four8ndqjtgllktwo4", "44our8ndqjtgllk2wo4"),
        ("jchmqgp85", "jchmqgp85"),
        ("dznstvthreeeightjzcxzsqbtsixqr8", "dznstv3hree8ightjzcxzsqbt6ixqr8"),
    ],
)
def test_convert_words_to_numbers(input_string, expected_result):
    assert convert_words_to_numbers(input_string) == expected_result


def test_process_file_a():
    result = process_file_a("../part_a_sample.txt")

    assert result == 142


def test_process_file_b():
    result = process_file_b("../part_b_sample.txt")

    assert result == 281


if __name__ == "__main__":
    pytest.main(sys.argv)
