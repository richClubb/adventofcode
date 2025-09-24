const std = @import("std");
const expect = std.testing.expect;

const seed_map = @import("day5").seed_map;
const SeedMap = seed_map.SeedMap;

pub const SeedMapLayer = struct {
    seed_maps: []SeedMap,

    pub fn add_map(self: SeedMapLayer, new_map: SeedMap) null {
        _ = self;
        _ = new_map;
    }

    pub fn map_seed(self: SeedMapLayer, seed_value: u64) ?u64 {
        _ = self;
        _ = seed_value;
        return null;
    }
};

pub const SeedMapLayers = struct {
    seed_map_layers: []SeedMapLayer,

    pub fn add_map_layer(self: SeedMapLayers, new_map_layer: SeedMapLayer) null {
        _ = self;
        _ = new_map_layer;
    }

    pub fn map_seed(self: SeedMapLayers, seed_value: u64) u64 {
        _ = self;
        _ = seed_value;
        return 0;
    }
};

test "map seed on layer" {
    var test_layer: SeedMapLayer = {};

    test_layer.add_map(SeedMap{ 2, 5, 5 });
    test_layer.add_map(SeedMap{ 10, 50, 2 });

    try expect(test_layer.map_seed(0) == null);
    try expect(test_layer.map_seed(1) == null);
    try expect(test_layer.map_seed(2) == 5);
    try expect(test_layer.map_seed(4) == 5);
    try expect(test_layer.map_seed(6) == 9);
    try expect(test_layer.map_seed(7) == null);

    try expect(test_layer.map_seed(9) == null);
    try expect(test_layer.map_seed(10) == 50);
    try expect(test_layer.map_seed(11) == 51);
    try expect(test_layer.map_seed(12) == null);
}
