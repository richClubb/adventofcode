# AoC 2015 - Day 09

https://adventofcode.com/2015/day/9

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Notes

This is a typical dykestra's argorithm, I think.

I did notice that routes can be reversed which reduced the dataset by half.

This is a perfect example of "kill your darlings" I went through about 4 iterations before landing on this one and despite hundreds of lines of code going in the bin I'm happy for it.

# Build / Run

## Build

```
cargo build [--release]
```

## Run

```
cargo run [path] [run]
```

or

```
cd [build dir]
./day11 [path] [run]
```

where `[run]` is:
* part_a
* part_b

# Improvements / research

* Not sure if there is a better way to do this with pattern matching
* I don't know if there is a way to use a generic algorithm for this