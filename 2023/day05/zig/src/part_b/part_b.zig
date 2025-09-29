const std = @import("std");

const seed_map = @import("seed_map");
const seed_map_layer = @import("seed_map_layer");
const seed_range = @import("seed_range");

fn extract_seed_ranges_from_string(seed_string: []const u8) error{ OutOfMemory, InvalidCharacter, Overflow }![]seed_range.SeedRange {
    const allocator = std.heap.page_allocator;
    const numbers_string = try std.mem.replaceOwned(u8, allocator, seed_string, "seeds: ", "");

    var numbers = try std.ArrayListUnmanaged(u64).initCapacity(allocator, 1);

    var number_strings = std.mem.splitScalar(u8, numbers_string, ' ');

    while (number_strings.next()) |number_string| {
        const number = try std.fmt.parseInt(u64, number_string, 10);
        try numbers.append(allocator, number);
    }

    var seed_ranges = try std.ArrayListUnmanaged(seed_range.SeedRange).initCapacity(allocator, numbers.items.len / 2);
    var index: usize = 0;
    while (index < numbers.items.len) : (index += 2) {
        //std.debug.print("One: {any} two {any}\n", .{ numbers.items[index], numbers.items[index + 1] });
        try seed_ranges.append(allocator, seed_range.SeedRange.init(numbers.items[index], numbers.items[index + 1]));
    }

    return seed_ranges.items;
}

pub fn part_b(file_path: []const u8) !u64 {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var file_buffer: [4096]u8 = undefined;
    var reader = file.reader(&file_buffer);

    var seed_ranges: []seed_range.SeedRange = undefined;
    const allocator = std.heap.page_allocator;
    var seedMapLayers = try std.ArrayListUnmanaged(seed_map_layer.SeedMapLayer).initCapacity(allocator, 1);
    var currSeedMapLayer: *seed_map_layer.SeedMapLayer = undefined;
    var index: u64 = 0;

    while (reader.interface.takeDelimiterExclusive('\n')) |line| {
        if (line.len == 0) {
            continue;
        }

        const seed_string = std.mem.indexOf(u8, line, "seeds: ");
        if (seed_string != null) {
            seed_ranges = try extract_seed_ranges_from_string(line);
            continue;
        }

        const seed_map_layer_name = std.mem.indexOf(u8, line, ":");
        if (seed_map_layer_name != null) {
            index = index + 1;
            try seedMapLayers.append(allocator, try seed_map_layer.SeedMapLayer.init());
            currSeedMapLayer = &seedMapLayers.items[index - 1];
            continue;
        }

        try currSeedMapLayer.add_map(try seed_map.SeedMap.init_from_string(line));
    } else |err| switch (err) {
        error.EndOfStream => {}, // Normal termination
        else => return err, // Propagate error
    }

    var min_value: u64 = std.math.maxInt(u64);
    for (seed_ranges) |aSeedRange| {
        const seed_range_start: u64 = aSeedRange.start;
        const seed_range_end: u64 = aSeedRange.end;
        for (seed_range_start..seed_range_end) |seed| {
            var value = seed;
            for (seedMapLayers.items) |aSeedMapLayer| {
                value = aSeedMapLayer.map_seed(value).?;
            }

            if (value < min_value) {
                min_value = value;
            }
        }
    }

    return min_value;
}

// test "read file" {
//     _ = part_a("/workspaces/adventofcode/2023/day05/sample_data.txt");
// }
