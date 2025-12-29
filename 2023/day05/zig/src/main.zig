const std = @import("std");
const zig_build = @import("zig_build");
const part_a = @import("part_a").part_a;
const part_b = @import("part_b").part_b;
const clap = @import("clap");

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();

    // First we specify what parameters our program can take.
    // We can use `parseParamsComptime` to parse a string into an array of `Param(Help)`.
    const params = comptime clap.parseParamsComptime(
        \\-h, --help             Display this help and exit.
        \\-f, --file <str>   An option parameter, which takes a value.
        \\-r, --run <str>    An option parameter which can be specified multiple times.
        \\
    );

    // Initialize our diagnostics, which can be used for reporting useful errors.
    // This is optional. You can also pass `.{}` to `clap.parse` if you don't
    // care about the extra information `Diagnostics` provides.
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = gpa.allocator(),
    }) catch |err| {
        // Report useful error and exit.
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0)
        std.debug.print("--help\n", .{});
    if (res.args.file == null) {
        std.debug.print("File must be specified\n", .{});
        return;
    }
    if (res.args.run == null) {
        std.debug.print("Run type must be specified, part_a, part_b\n", .{});
        return;
    }

    std.debug.print("Advent of code 2023 Day 05\n", .{});
    if (std.mem.eql(u8, res.args.run orelse "", "part_a")) {
        std.debug.print("part_a result: {!}\n", .{part_a(res.args.file.?)});
    } else if (std.mem.eql(u8, res.args.run orelse "", "part_b")) {
        std.debug.print("part_b result: {!}\n", .{part_b(res.args.file.?)});
    }
}
