const std = @import("std");

fn extract_numbers(allocator: *const std.mem.Allocator, input: []const u8) ![]usize {
    var output = try allocator.alloc(usize, 3);
    var output_index: usize = 0;
    // var num_buffer: [5]u8 = undefined;
    // var num_buffer_index: usize = 0;

    var number: usize = 0;

    // @memset(&num_buffer, 0);

    for (input) |character| {
        if (' ' == character) {
            if (0 != number) {
                output[output_index] = number;
                output_index += 1;
            }
            number = 0;
            continue;
        }

        const next_digit = (character - 48);
        number = (number * 10) + (next_digit);
    }
    output[output_index] = number;

    return output;
}

pub fn part_a(file_path: []const u8) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var reader = file.reader(&file_buffer);

    const allocator = std.heap.page_allocator;

    var successes: usize = 0;

    while (reader.interface.takeDelimiterExclusive('\n')) |line| {
        // std.debug.print("{s}\n", .{line});
        const result = try extract_numbers(&allocator, line);
        const side_1 = result[0];
        const side_2 = result[1];
        const side_3 = result[2];

        const comp_1 = (side_1 + side_2) > side_3;
        const comp_2 = (side_1 + side_3) > side_2;
        const comp_3 = (side_2 + side_3) > side_1;

        if ((comp_1 & comp_2 & comp_3) == true) {
            successes += 1;
            // std.debug.print("Triangle '{s}' succeeded\n", .{line});
        } else {
            // std.debug.print("Triangle '{s}' failed\n", .{line});
        }
    } else |err| switch (err) {
        error.EndOfStream => {}, // Normal termination
        else => return err, // Propagate error
    }

    std.debug.print("Result: '{}'\n", .{successes});
}
