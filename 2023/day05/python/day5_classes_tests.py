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
        [(1, 5, 2, 1, 1, None), (1, 5, 2, 5, 1, Seed(1))],
    )
    def test_mapping_seed(
        self, dest, src, size, seed_value, expected_value, expected_result
    ):
        mapping = Mapping(dest, src, size)
        seed = Seed(seed_value)

        assert mapping.map_seed(seed) == expected_result

    @pytest.mark.parametrize(
        "dest, src, size, seed_range, expected_result",
        [
            (1, 5, 2, SeedRange(1, 2), ([SeedRange(1, 2)], None)),
            (1, 5, 2, SeedRange(8, 9), ([], SeedRange(8, 9))),
            (
                1,
                5,
                2,
                SeedRange(3, 3),
                ([SeedRange(3, end=4), SeedRange(1, end=1)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(3, 4),
                ([SeedRange(3, end=4), SeedRange(1, end=2)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(3, 6),
                ([SeedRange(3, end=4), SeedRange(1, end=4)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(3, 7),
                ([SeedRange(3, end=4), SeedRange(1, end=4)], SeedRange(9, end=9)),
            ),
            (
                1,
                5,
                4,
                SeedRange(5, 2),
                ([SeedRange(1, end=2)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(5, end=8),
                ([SeedRange(1, end=4)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(5, end=10),
                ([SeedRange(1, end=4)], SeedRange(9, 2)),
            ),
            (
                1,
                5,
                4,
                SeedRange(6, end=7),
                ([SeedRange(2, end=3)], None),
            ),
            (
                1,
                5,
                4,
                SeedRange(6, end=8),
                ([SeedRange(2, end=4)], None),
            ),
        ],
    )
    def test_mapping_seed_range(self, dest, src, size, seed_range, expected_result):
        mapping = Mapping(dest, src, size)
        result = mapping.map_seed_range(seed_range)
        assert result == expected_result


class TestMappingLayer:
    @pytest.mark.parametrize(
        "maps, seed, expected_value",
        [
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(1), None),
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(5), Seed(1)),
            ([Mapping(1, 5, 2), Mapping(7, 10, 3)], Seed(11), Seed(8)),
        ],
    )
    def test_mapping_layer_map_seed(self, maps, seed, expected_value):
        mapping_layer = MappingLayer()
        for map in maps:
            mapping_layer.add_mapping(map)

        assert mapping_layer.map_seed(seed) == expected_value

    @pytest.mark.parametrize(
        "maps, seed_ranges, expected_value",
        [
            (
                [Mapping(1, 5, 2), Mapping(7, 10, 3)],
                [SeedRange(1, end=4)],
                [SeedRange(1, end=4)],
            ),
            (
                [Mapping(1, 5, 2), Mapping(7, 10, 3)],
                [SeedRange(3, end=6)],
                [SeedRange(1, end=2), SeedRange(3, end=4)],
            ),
            (
                [Mapping(1, 5, 2), Mapping(16, 10, 3)],
                [SeedRange(3, end=15)],
                [
                    SeedRange(1, end=2),
                    SeedRange(3, end=4),
                    SeedRange(7, end=9),
                    SeedRange(13, end=15),
                    SeedRange(16, end=18),
                ],
            ),
        ],
    )
    def test_mapping_layer_map_range(self, maps, seed_ranges, expected_value):
        mapping_layer = MappingLayer()
        for map in maps:
            mapping_layer.add_mapping(map)

        result = mapping_layer.map_seed_ranges(seed_ranges)

        assert result == expected_value


class TestMappingLayers:
    @pytest.mark.parametrize(
        "mapping_lists, seed, expected_value",
        [
            (
                [[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(12, 20, 5)]],
                Seed(1),
                Seed(1),
            ),
            (
                [[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(12, 20, 5)]],
                Seed(5),
                Seed(1),
            ),
            (
                [[Mapping(1, 5, 2), Mapping(7, 10, 3)], [Mapping(20, 5, 5)]],
                Seed(11),
                Seed(23),
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

        result = mapping_layers.map_seed(seed)

        assert result == expected_value


if __name__ == "__main__":
    pytest.main(sys.argv)
