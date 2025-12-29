const std = @import("std");
const expect = std.testing.expect;

const seed_map = @import("seed_map");
const seed_range = @import("seed_range");

pub const SeedMapLayer = struct {
    seed_maps: []seed_map.SeedMap,
    name: []u8,

    const Self = @This();
    pub fn init() error{OutOfMemory}!SeedMapLayer {
        const allocator = std.heap.page_allocator;
        return SeedMapLayer{
            .name = undefined,
            .seed_maps = try allocator.alloc(seed_map.SeedMap, 1),
        };
    }

    pub fn add_map(self: *Self, seedMap: seed_map.SeedMap) error{OutOfMemory}!void {
        const allocator = std.heap.page_allocator;
        var new_seed_maps = try allocator.alloc(seed_map.SeedMap, self.seed_maps.len + 1);

        for (self.seed_maps, 0..) |a_seed_map, index| {
            new_seed_maps[index] = a_seed_map;
        }

        new_seed_maps[self.seed_maps.len] = seedMap;

        self.seed_maps = new_seed_maps;
    }

    pub fn map_seed(self: SeedMapLayer, value: u64) ?u64 {
        for (self.seed_maps) |aSeedMap| {
            const new_value = aSeedMap.map_seed(value);
            if (new_value != null) {
                return new_value;
            }
        }
        return value;
    }
};

test "Init seed map layer" {}

test "add map to layer" {}

test "map seed in layer" {
    var test_seed_map_layer = try SeedMapLayer.init();
    test_seed_map_layer.add_map(seed_map.SeedMap.init(2, 10, 5));
    test_seed_map_layer.add_map(seed_map.SeedMap.init(20, 50, 2));

    try expect(test_seed_map_layer.map_seed(1) == 1);
    try expect(test_seed_map_layer.map_seed(2) == 10);
    try expect(test_seed_map_layer.map_seed(4) == 12);
    try expect(test_seed_map_layer.map_seed(7) == 7);

    try expect(test_seed_map_layer.map_seed(19) == 19);
    try expect(test_seed_map_layer.map_seed(20) == 50);
    try expect(test_seed_map_layer.map_seed(21) == 51);
    try expect(test_seed_map_layer.map_seed(22) == 22);
}

test "map seed in layers" {}

test "find min value in range" {}
