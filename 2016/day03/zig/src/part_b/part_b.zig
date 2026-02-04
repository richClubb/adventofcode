const std = @import("std");

fn extract_numbers(allocator: *const std.mem.Allocator, input: []const u8) ![]usize {
    var output = try allocator.alloc(usize, 3);
    var output_index: usize = 0;

    var number: usize = 0;

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

fn is_triangle(side_a: usize, side_b: usize, side_c: usize) bool {
    const comp_1 = (side_a + side_b) > side_c;
    const comp_2 = (side_a + side_c) > side_b;
    const comp_3 = (side_b + side_c) > side_a;

    if ((comp_1 & comp_2 & comp_3) == true) {
        // std.debug.print("Triangle '{} {} {}' succeeded\n", .{ side_a, side_b, side_c });
        return true;
    } else {
        // std.debug.print("Triangle '{} {} {}' failed\n", .{ side_a, side_b, side_c });
        return false;
    }
}

pub fn part_b(file_path: []const u8) !void {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var reader = file.reader(&file_buffer);

    const allocator = std.heap.page_allocator;

    var number_list = try std.ArrayList([]usize).initCapacity(allocator, 10);

    var successes: usize = 0;

    while (reader.interface.takeDelimiterExclusive('\n')) |line| {
        // std.debug.print("{s}\n", .{line});
        const result = try extract_numbers(&allocator, line);

        try number_list.append(allocator, result);
    } else |err| switch (err) {
        error.EndOfStream => {}, // Normal termination
        else => return err, // Propagate error
    }

    var index: usize = 0;
    while (index < number_list.items.len) : (index += 3) {
        for (0..3) |triangle_index| {
            const side_a = number_list.items[index][triangle_index];
            const side_b = number_list.items[index + 1][triangle_index];
            const side_c = number_list.items[index + 2][triangle_index];

            if (is_triangle(side_a, side_b, side_c)) {
                successes += 1;
            }
        }
        // std.debug.print("{} {} {}\n", .{ number[0], number[1], number[2] });
    }

    std.debug.print("Result: '{}'\n", .{successes});
}
