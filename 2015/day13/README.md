# AoC 2015 - Day 13

https://adventofcode.com/2015/day/13

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
./day13 [path]
```

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

I got a set of permutations but didn't manage to reduce at all, I might be able to do so by looking for reversals but it's a bit of a PITA for an algorithm that runs in a few seconds.

# Improvements

* Remove duplicates
* Improve datastructure
* Improve mechanism for finding the permutations
