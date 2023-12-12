#!/bin/env python3
import pytest
from day2 import (
    part_a,
    part_b,
    is_game_possible,
    is_round_possible,
    check_red,
    check_green,
    check_blue,
    cubes_in_round,
    max_cubes_in_game,
)


@pytest.mark.parametrize(
    "input_round, expected_result",
    [
        ("3blue,4red", (4, 0, 3)),
        ("1red,2green,6blue", (1, 2, 6)),
        ("2green", (0, 2, 0)),
        ("8green,6blue,20red;", (20, 8, 6)),
        ("5blue,4red,13green", (4, 13, 5)),
        ("5green,1red", (1, 5, 0)),
    ],
)
def test_cubes_in_round(input_round, expected_result):
    assert cubes_in_round(input_round) == expected_result


@pytest.mark.parametrize(
    "input_game, expected_result",
    [
        ("3blue,4red;1red,2green,6blue;2green", (4, 2, 6)),
        ("8green,6blue,20red;5blue,4red,13green;5green,1red", (20, 13, 6)),
    ],
)
def test_max_cubes_in_game(input_game, expected_result):
    assert max_cubes_in_game(input_game) == expected_result


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


def test_part_b():
    result = part_b("part_a_sample.txt")

    assert result == 2286
