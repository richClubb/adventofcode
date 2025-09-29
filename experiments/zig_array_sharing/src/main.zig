const std = @import("std");
const zig_array_sharing = @import("zig_array_sharing");

pub fn share_array(size: u64) ![]u64 {
    const allocator = std.heap.page_allocator;
    var numbers = try std.ArrayListUnmanaged(u64).initCapacity(allocator, 1);

    for (0..size) |i| {
        try numbers.append(allocator, i);
    }

    return numbers.items;
}

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    const numbers = try share_array(5);

    for (numbers) |number| {
        std.debug.print("Number: {any}\n", .{number});
    }
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
