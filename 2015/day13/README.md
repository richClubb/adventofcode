# AoC 2015 - Day 13

https://adventofcode.com/2015/day/13

# Status

* Part A - Incomplete
* Part B - Incomplete

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

# Notes

This is a [circle permutation](https://ilovemaths.com/3permcirc.asp) problem so the number of possible unique combinations is

```
(n-1)! / 2 
```

Where n is the number of people. So for the sample data we have 4 people `(4-1)!/2 = 3!/2 = 6/2 = 3` for the full data we have 8 people `(8-1)!/2 = 7!/2 = 5040/2 = 2520`.

```
    a
  b   c
    d
```

is the same as

```
    c
  a   d
    b
```

How do we find out those permutations? 

# Improvements

