# AoC 2015 - Day 18

https://adventofcode.com/2015/day/18

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
cargo run [input_file] [steps] [run]
```

or

```
cd [build dir]
./day18 [input_file] [steps] [run]
```

where `[run]` is:
* part_a
* part_b

# Notes

This looks like conways game of life (https://conwaylife.com/)

# Improvements

* Had an idea to do this by adding the neighbour pixels to each pixel rather than having to calculate which pixels they were but ran up against the Rust borrow checker and stuff, would like to revisit this.

