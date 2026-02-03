const std = @import("std");

fn extract_numbers(allocator: *const std.mem.Allocator, input: []const u8) ![]usize {
    var output = try allocator.alloc(usize, 3);
    var output_index: usize = 0;
    var num_buffer: [5]u8 = undefined;
    var num_buffer_index: usize = 0;
    @memset(&num_buffer, 0);

    for (input) |character| {
        if (' ' == character) {
            if (num_buffer_index != 0) {
                std.debug.print("  {s}\n", .{num_buffer});
                const number = try std.fmt.parseInt(usize, &num_buffer, 10);
                output[output_index] = number;
                output_index += 1;

                // clear out for next number
                @memset(&num_buffer, 0);
                num_buffer_index = 0;
            }
            continue;
        }
        num_buffer[num_buffer_index] = character;
        num_buffer_index += 1;
    }
    std.debug.print("  {s}\n", .{num_buffer});

    std.debug.print("\n", .{});

    return output;
}

pub fn part_a(file_path: []const u8) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var reader = file.reader(&file_buffer);

    const allocator = std.heap.page_allocator;

    while (reader.interface.takeDelimiterExclusive('\n')) |line| {
        // std.debug.print("{s}\n", .{line});
        const result = try extract_numbers(&allocator, line);
        std.debug.print("{} {} {}", .{ result[0], result[1], result[2] });
    } else |err| switch (err) {
        error.EndOfStream => {}, // Normal termination
        else => return err, // Propagate error
    }
}
