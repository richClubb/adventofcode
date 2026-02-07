# AoC 2016 - Day 03

https://adventofcode.com/2016/day/03

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Build / Run

## Build

from the `zig/` directory

```
zig build
```

## Run

from the `zig/` directory

```
zig build run -- -f [path] -r [run]
```

or 

```
zig build
./zig-out/bin/zig -f [path] -r [run]
```

Where `[run]` is:
* part_a
* part_b

# Notes

I really like that while loops can have an index increment 

```
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
```

# Improvements

* I need to figure out a nicer regex approach, I didn't want to go through the process of importing `regex.h`
* I want a nicer parsing int approach. Mine works but is stupid

# Time

* Solution took about 1 - 2 hours, there is a lot of stuff about the slicing and allocation that I don't fully understand. Need to look into it.
* Setup took about 1 - 2 hours to get used to some of the toolchain again.
