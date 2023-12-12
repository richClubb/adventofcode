#!/bin/env python3
import pytest


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
        ("one", "1"),
        ("1", "1"),
        ("one1", "11"),
        ("nineninenine", "999"),
        ("one1zzztwo8", "11zzz28"),
        ("nine57", "957"),
        ("7rdtplhbvddvlkonefqsttj", "7rdtplhbvddvlk1fqsttj"),
        ("four6mssqzhgxt", "46mssqzhgxt"),
        ("gphnqxhlhzzftghk767", "gphnqxhlhzzftghk767"),
        ("mbcbpjcsnt4six", "mbcbpjcsnt46"),
        ("one9xmhvzklmzffive1kcsixmnsbm2", "19xmhvzklmzf51kc6mnsbm2"),
        ("1dgschj", "1dgschj"),
        ("nine8foursnczninednds", "984sncz9dnds"),
        (
            "9sevensixrsrgmclkvthkgtxqtwovtlxfrdnllxvsghslh",
            "976rsrgmclkvthkgtxq2vtlxfrdnllxvsghslh",
        ),
        ("seven443six8three31", "744368331"),
        ("pppmfmnfourtworxrqrfhbgx8vvxgrjzhvqmztltwo", "pppmfmn42rxrqrfhbgx8vvxgrjzhvqmztl2"),
        ("56oneninethreevv4chvlfljbrthree","56193vv4chvlfljbr3"),
        ("9sixseven9zspvdsqxzf","9679zspvdsqxzf"),
        ("4four8ndqjtgllktwo4","448ndqjtgllk24"),
        ("jchmqgp85","jchmqgp85"),
        ("dznstvthreeeightjzcxzsqbtsixqr8", "dznstv38jzcxzsqbt6qr8"),
    ],
)
def test_convert_words_to_numbers(input_string, expected_result):
    assert convert_words_to_numbers(input_string) == expected_result


def test_process_file_a():
    result = process_file_a("part_a_sample.txt")

    assert result == 142


def test_process_file_b():
    result = process_file_b("part_b_sample.txt")

    assert result == 281
