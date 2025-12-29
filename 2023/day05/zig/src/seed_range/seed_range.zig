const std = @import("std");
const expect = std.testing.expect;

pub const SeedRange = struct {
    start: u64,
    end: u64,
    size: u64,

    pub fn init(start: u64, size: u64) SeedRange {
        return SeedRange{
            .start = start,
            .size = size,
            .end = (start + size),
        };
    }

    pub fn init_from_string(input_string: []const u8) error{ OutOfMemory, Overflow, InvalidCharacter }!SeedRange {
        const allocator = std.heap.page_allocator;
        var numbers = try std.ArrayListUnmanaged(u64).initCapacity(allocator, 2);
        var number_strings = std.mem.splitScalar(u8, input_string, ' ');
        while (number_strings.next()) |number_string| {
            const number = try std.fmt.parseInt(u64, number_string, 10);
            try numbers.append(allocator, number);
        }
        if (numbers.items.len != 2) {
            return error.InvalidCharacter;
        }

        return SeedRange.init(numbers.items[0], numbers.items[1]);
    }
};

test "seed map parse from list" {
    const test_range_1: SeedRange = try SeedRange.init_from_string("1 10");

    try expect(test_range_1.start == 1);
    try expect(test_range_1.end == 11);
    try expect(test_range_1.size == 10);

    const test_range_2: SeedRange = try SeedRange.init_from_string("1000000 5000000");

    try expect(test_range_2.start == 1000000);
    try expect(test_range_2.end == 6000000);
    try expect(test_range_2.size == 5000000);
}
