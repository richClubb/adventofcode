# AoC 2015 - Day 20

https://adventofcode.com/2015/day/20

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
cargo run [input] [run]
```

or

```
cd [build dir]
./day20 [input] [run]
```

where `[run]` is:
* part_a
* part_b

# Notes

* brute force is a silly way to do this.
  * sum of prime factors is probably the sensible way
  * https://math.stackexchange.com/questions/163245/finding-sum-of-factors-of-a-number-using-prime-factorization

for part_a it would be figuring out the sum of primes for each number rather than having to iterate over large integers

for part_b it would be the same but you'd have to figure out the values lower than the max number of houses and subtract it.

I'm still thinking that you can cache / store the primes at the beginning so you don't have to keep re-calculating them.

# Improvements
