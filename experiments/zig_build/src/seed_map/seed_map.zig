const std = @import("std");
const expect = std.testing.expect;

pub const SeedMap = struct {
    source: u64,
    target: u64,
    size: u64,

    pub fn init(source: u64, target: u64, size: u64) SeedMap {
        return SeedMap{
            .source = source,
            .target = target,
            .size = size,
        };
    }

    // to do
    pub fn init_from_string(input_string: []const u8) error{ OutOfMemory, Overflow, InvalidCharacter }!SeedMap {
        const allocator = std.heap.page_allocator;
        var numbers = try std.ArrayListUnmanaged(u64).initCapacity(allocator, 3);
        var number_strings = std.mem.splitScalar(u8, input_string, ' ');
        while (number_strings.next()) |number_string| {
            const number = try std.fmt.parseInt(u64, number_string, 10);
            try numbers.append(allocator, number);
        }
        if (numbers.items.len != 3) {
            return error.InvalidCharacter;
        }

        return SeedMap{ .source = numbers.items[1], .target = numbers.items[0], .size = numbers.items[2] };
    }

    pub fn map_seed(self: SeedMap, value: u64) ?u64 {
        if ((value >= self.source) and
            (value < (self.source + self.size)))
        {
            return (value - self.source) + self.target;
        }
        return null;
    }
};

test "map seed" {
    const test_map: SeedMap = .{
        .source = 5,
        .target = 10,
        .size = 5,
    };

    try expect(test_map.map_seed(1) == null);
    try expect(test_map.map_seed(2) == null);
    try expect(test_map.map_seed(4) == null);
    try expect(test_map.map_seed(5) == 10);
    try expect(test_map.map_seed(6) == 11);
    try expect(test_map.map_seed(7) == 12);
    try expect(test_map.map_seed(8) == 13);
    try expect(test_map.map_seed(9) == 14);
    try expect(test_map.map_seed(10) == null);
}

test "seed map parse from list" {
    const test_map: SeedMap = try SeedMap.init_from_string("1 10 5");

    try expect(test_map.source == 10);
    try expect(test_map.target == 1);
    try expect(test_map.size == 5);
}
