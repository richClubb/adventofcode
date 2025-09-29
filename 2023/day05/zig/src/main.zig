const std = @import("std");
const zig_build = @import("zig_build");
const part_a = @import("part_a").part_a;
const part_b = @import("part_b").part_b;

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.

    //const file_path = "/workspaces/adventofcode/2023/day05/sample_data.txt";
    const file_path = "/workspaces/adventofcode/2023/day05/full_data.txt";
    std.debug.print("part_a result: {!}\n", .{part_a(file_path)});

    std.debug.print("part_b result: {!}\n", .{part_b(file_path)});
}
