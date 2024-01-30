#!/bin/env python3

from day5_classes import Seed, SeedRange, Mapping, MappingLayer, MappingLayers
import pytest
import sys


class TestSeed:
    @pytest.mark.parametrize(
        "seed_value, expected_value",
        [(1, 1), (2, 2), (1000099, 1000099)],
    )
    def test_seed_range_init(self, seed_value, expected_value):
        seed = Seed(seed_value)

        assert seed.value == expected_value

    @pytest.mark.parametrize(
        "seed_value, comparison_value, expected_result",
        [(1, 2, True), (1, 1, False)],
    )
    def test_seed_less_than(self, seed_value, comparison_value, expected_result):
        seed = Seed(seed_value)
        comp_seed = Seed(comparison_value)

        assert (seed < comparison_value) == expected_result
        assert (seed < comp_seed) == expected_result

    # Need to do other comparisons

    @pytest.mark.parametrize(
        "seed_value, new_value",
        [(1, 2), (1, 1), (1, 7)],
    )
    def test_seed_less_than(self, seed_value, new_value):
        seed = Seed(seed_value)
        seed.value = new_value

        assert seed.value == new_value


class TestSeedRange:
    @pytest.mark.parametrize(
        "seed_start, seed_size, expected_value",
        [(1, 2, (1, 2)), (2, 4, (2, 5))],
    )
    def test_seed_range_init(self, seed_start, seed_size, expected_value):
        seed_range = SeedRange(seed_start, seed_size)

        assert seed_range.value == expected_value

    @pytest.mark.parametrize(
        "seed_start, seed_size, comparison_value, expected_result",
        [(1, 2, (1, 2), True), (1, 2, (1, 4), False)],
    )
    def test_seed_range_equal(
        self, seed_start, seed_size, comparison_value, expected_result
    ):
        seed_range = SeedRange(seed_start, seed_size)
        comp_seed_range = SeedRange(comparison_value[0], comparison_value[1])

        assert (seed_range == comparison_value) == expected_result
        assert (seed_range == comp_seed_range) == expected_result

    @pytest.mark.parametrize(
        "seed_start, seed_size, expected_result",
        [(1, 2, [1, 2]), (2, 4, [2, 3, 4, 5])],
    )
    def test_seed_range_init(self, seed_start, seed_size, expected_result):
        seed_range = SeedRange(seed_start, seed_size)

        result = []
        for seed in seed_range:
            result.append(seed)

        assert result == expected_result


class TestMapping:
    @pytest.mark.parametrize(
        "dest, src, size, seed_value, expected_value, expected_result",
        [(1, 5, 2, 1, 1, False), (1, 5, 2, 5, 1, True)],
    )
    def test_mapping(
        self, dest, src, size, seed_value, expected_value, expected_result
    ):
        mapping = Mapping(dest, src, size)
        seed = Seed(seed_value)

        assert mapping.map_seed(seed) == expected_result
        assert seed.value == expected_value


class TestMappingLayer:
    @pytest.mark.parametrize(
        "maps, seed, expected_value, expected_result",
        [
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(1), 1, False),
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(5), 1, True),
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(11), 8, True),
        ],
    )
    def test_mapping_layer(self, maps, seed, expected_value, expected_result):
        mapping_layer = MappingLayer()
        for map in maps:
            mapping_layer.add_mapping(map)

        assert mapping_layer.map_seed(seed) == expected_result
        assert seed.value == expected_value


class TestMappingLayers:
    @pytest.mark.parametrize(
        "mapping_lists, seed, expected_value",
        [
            ([[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(12, 20, 5)]], Seed(1), 1),
            ([[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(12, 20, 5)]], Seed(5), 1),
            (
                [[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(20, 5, 5)]],
                Seed(11),
                23,
            ),
        ],
    )
    def test_mapping_layers(self, mapping_lists, seed, expected_value):
        mapping_layers = MappingLayers()

        for mapping_list in mapping_lists:
            mapping_layer = MappingLayer()
            for mapping in mapping_list:
                mapping_layer.add_mapping(mapping)
            mapping_layers.add_mapping_layer(mapping_layer)

        mapping_layers.map_seed(seed)

        assert seed.value == expected_value


if __name__ == "__main__":
    pytest.main(sys.argv)
