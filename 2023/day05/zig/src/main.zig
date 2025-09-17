const std = @import("std");
const fs = std.fs;
const print = std.debug.print;

pub fn main() !void {
    std.debug.print("Hello {s}\n", .{"World"});
}
