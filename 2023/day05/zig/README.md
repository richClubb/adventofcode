Advent of code 2023 Day 05 - Zig
--------------------------------

# Overview

I have mixed feeling about Zig...

The low level nature is nice, I like the control that it gives you. I don't mind the syntax but do feel it's pretty verbose. The fact that the language is still pre-release is a bit frustrating. Some of the guides were based on Zig 0.14 and things changed in 0.15, this was minor but annoying.

Allocators are a minefield, I nedd to spend some more time working on them and getting to learn some of the patterns, the fact thet you can have different allocators for different purposes is cool but I don't know when I'd need to use that right now. Knowing the patterns behind when and where you might want to pass in an allocator into a function is a bit of a nightmare.

The performance optimisaiton was also a frustrating sticking point. Zig was MILES slower than everything else when in debug mode. It took over 4 hours on an unoptimised build, but optimised it was up there with Rust and C. 

The `build.zig` is another thing that needs investigation. It looks very capable and I think is very granular but I need to spend a proper few days researching the different features and exactly how to use it.

# Build / Run

```
zig build
./zig-out/bin/day5 -f [path] -r [run]
```

Where [run] is:
* part_a
* part_b

make sure to remember to use the `zig build -Doptimize=ReleaseFast` flag or it takes an eternity.

# Test

```
zig test [path]
```