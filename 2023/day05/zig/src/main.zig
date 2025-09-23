const std = @import("std");

const seed_map = @import("seed_map/seed_map.zig");

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    try std.fs.File.stdout().writeAll("Advent of code 2023 - Day 5\n");

    const seed_map_1: seed_map.SeedMap = .{ .source = 5, .target = 10, .size = 5 };

    std.debug.print("Start: {?}\n", .{seed_map_1.map_seed(4)});
}
