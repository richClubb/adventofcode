const std = @import("std");

const seed_map_layer = @import("seed_map_layer");

pub fn part_a(_: []const u8) u64 {
    // const allocator = std.heap.page_allocator;

    // // Open the file
    // var file = try std.fs.cwd().openFile(file_path, .{});
    // defer file.close();

    // // Read the file contents
    // const contents = try file.readToEndAlloc(allocator, 2048);

    // defer allocator.free(contents);

    // // Print the file contents
    // std.debug.print("{s}", .{contents});

    return 0;
}

test "read file" {
    _ = part_a("/workspaces/adventofcode/2023/day05/sample_data.txt");
}
