const seed_map = @import("seed_map");
const seed_range = @import("seed_range");

pub const SeedMapLayer = struct {
    seed_maps: []seed_map.SeedMap,
    name: []u8,

    pub fn init() SeedMapLayer {
        return SeedMapLayer{};
    }

    pub fn add_map(_: seed_map.SeedMap) void {}

    pub fn map_seed(_: u64) u64 {
        return 0;
    }
};

pub const SeedMapLayers = struct {
    seed_map_layers: []SeedMapLayer,

    pub fn map_seed(_: u64) u64 {
        return 0;
    }

    pub fn min_in_seed_range(_: seed_range.SeedRange) u64 {}
};

test "Init seed map layer" {}

test "add map to layer" {}

test "map seed in layer" {}

test "map seed in layers" {}

test "find min value in range" {}
