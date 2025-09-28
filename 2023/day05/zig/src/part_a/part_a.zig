const std = @import("std");

const seed_map_layer = @import("seed_map_layer");

pub fn part_a(file_path: []const u8) !u64 {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var reader = file.reader(&file_buffer);
    while (reader.interface.takeDelimiterExclusive('\n')) |line| {
        std.debug.print("{s}\n", .{line});
    } else |err| switch (err) {
        error.EndOfStream => {}, // Normal termination
        else => return err, // Propagate error
    }

    return 0;
}

// test "read file" {
//     _ = part_a("/workspaces/adventofcode/2023/day05/sample_data.txt");
// }
