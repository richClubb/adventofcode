# AoC 2015 - Day 19

https://adventofcode.com/2015/day/19

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Build / Run

## Build

```
cargo build [--release]
```

## Run

```
cargo run [path]
```

or

```
cd [build dir]
./day19 [path] [run]
```


where `[run]` is:
* part_a
* part_b

# Notes

For part_b it was better to solve it backwards. I though this might be the case but I did look up someone else's analysis for this.

# Improvements

Just realised why my part_b is failing. It's replacing all of the instances but should only be replacing the first. Yep that worked.
