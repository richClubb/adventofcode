# AoC 2015 - Day 01

https://adventofcode.com/2015/day/1

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
cargo run [path] [run]
```

or

```
cd [build dir]
./day01 [path] [run]
```

where `[run]` is:
* part_a
* part_b

# Notes

Took about 20 minutes to do a very basic version. It doesn't have any test cases or error checking and is a pretty nested for loop

There isn't much too this problem, you essentially have to keep track of the number of `(` and `)` you've encountered and increment / decrement a value. I don't / can't think of a more intelligent way to do this that isn't some kind of iterator, whether it's a loop or a `fold` or something similar.

# Improvements

* Could this be done without having a mutable variable?
  * part a - Done
* Could this be done without the for loops?
  * part a - Done

It's a little hard to read but it works

```Rust
let file_total = buf_reader.lines().fold(0, |file_acc: i64, line| {
        file_acc + match line {
                Ok(x) => {
                x.chars().fold(0, |line_acc: i64, b: char| 
                    {
                        line_acc + match b {
                            '(' => 1,
                            ')' => -1,
                            _ => 0,
                        }
                    }
                )
            },
            Err(x) => 0,
        }
    }
);
```