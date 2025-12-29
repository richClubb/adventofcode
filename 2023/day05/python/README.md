Advent of code 2023 Day 05 - Python
-----------------------------------

# Overview

This was the first implementation which was done as a group as an engineering exercise at work. It took us about 30 minutes to come up with both the implementations but the Part B took hours to actually get a result, which is part of what spurred on this entire exercise.

It was very easy to do, I experimented with the 'inverse' search implementation which took only a few minutes to code up and got a result much faster but obviously that is more luck than science.

It was the only implementation to try calculating the range translation rather than just brute forcing the problem as the data manipulation was pretty simple but I could probably make it significantly more succinct if I tried. I wouldn't mind trying this in Rust.

# Build / Run

No building required, no third party libraries just pure old fashioned python.

## day5.py

Single threaded

```bash
./day5.py [path] [run]
```

where [run] is:
* part_a
* part_b_forward
* part_b_inverse

## day5_parallel.py

Multi threaded version

```bash
./day5_parallel.py [path] [run]
```

where [run] is:
* part_a
* part_b_forward
* part_b_inverse

## day5_ranges.py

This is the only current implementation on doing the translation on the ranges of numbers and not the individual numbers themselves. This is by far the fastest implementation.

```bash
./day5_ranges.py [path] [run]
```

where [run] is:
* part_a
* part_b_ranges

## day5_classes.py

This is an all in one implementation which includes single threaded part a, b and b inverse as well as the ranges calculation.

```bash
./day5_ranges.py [path] [run]
```

where [run] is:
* part_a
* part_b_forward
* part_b_inverse
* part_b_ranges

# Tests

There are various tests but I haven't put a lot of effort into making the testing infra solid. Have a look.

# To/do

Redevelop this into a more cohesive solution with a single entry point for all, and use poetry properly to run the program.