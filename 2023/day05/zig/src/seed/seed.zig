const std = @import("std");
const expect = std.testing.expect;

pub fn extract_seeds(input_string: []const u8) ![]u64 {
    const allocator = std.heap.page_allocator;
    var numbers = try std.ArrayListUnmanaged(u64).initCapacity(allocator, 2);
    var number_strings = std.mem.splitScalar(u8, input_string, ' ');
    while (number_strings.next()) |number_string| {
        const number = try std.fmt.parseInt(u64, number_string, 10);
        try numbers.append(allocator, number);
    }

    return numbers;
}

test "text extract seeds" {
    const numbers = try extract_seeds("1 2 4 8 16");

    try expect(numbers.len == 5);

    try expect(numbers.items[0] == 1);
    try expect(numbers.items[1] == 2);
    try expect(numbers.items[2] == 4);
    try expect(numbers.items[3] == 8);
    try expect(numbers.items[4] == 16);
}
