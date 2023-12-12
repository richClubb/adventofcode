#!/bin/env python3
import pytest
from day2 import part_a, is_game_possible, is_round_possible, check_red, check_green, check_blue

@pytest.mark.parametrize(
    "input_round, expected_result",
    [
        ("3blue,4red", True),
        ("1red,2green,6blue", True),
        ("2green", True),
        ("8green,6blue,20red;", False),
        ("5blue,4red,13green", True),
        ("5green,1red", True),
    ],
)
def test_check_red(input_round, expected_result):
    assert check_red(input_round) == expected_result

@pytest.mark.parametrize(
    "input_round, expected_result",
    [
        ("3blue,4red", True),
        ("1red,2green,6blue", True),
        ("2green", True),
        ("8green,6blue,20red;", True),
        ("5blue,4red,14green", False),
        ("5green,1red", True),
    ],
)
def test_check_green(input_round, expected_result):
    assert check_green(input_round) == expected_result

@pytest.mark.parametrize(
    "input_round, expected_result",
    [
        ("3blue,4red", True),
        ("1red,2green,6blue", True),
        ("2green", True),
        ("8green,6blue,20red;", True),
        ("15blue,4red,13green", False),
        ("5green,1red", True),
    ],
)
def test_check_blue(input_round, expected_result):
    assert check_blue(input_round) == expected_result


@pytest.mark.parametrize(
    "input_round, expected_result",
    [
        ("3blue,4red", True),
        ("1red,2green,6blue", True),
        ("2green", True),
        ("8green,6blue,20red;", False),
        ("5blue,4red,13green", True),
        ("5green,1red", True),
    ],
)
def test_is_round_possible(input_round, expected_result):
    assert is_round_possible(input_round) == expected_result


@pytest.mark.parametrize(
    "input_game, expected_result",
    [
        ("3blue,4red;1red,2green,6blue;2green", True),
        ("8green,6blue,20red;5blue,4red,13green;5green,1red", False),
    ],
)
def test_is_game_possible(input_game, expected_result):
    assert is_game_possible(input_game) == expected_result


def test_part_a():

    result = part_a("part_a_sample.txt")

    assert result == 8