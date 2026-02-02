# AoC 2015 - Day 05

https://adventofcode.com/2015/day/5

# Status

* Part A - COMPLETE
* Part B - COMPLETE

# Notes

Not including the 'z' in the original caught me out

Original
```Rust
    for character in 'a'..'z' {
        let match_string = format!("{0}{0}", character);

        if input.contains(&match_string)
        {
            return true;
        }
    }
```

Fixed
```Rust
    for character in 'a'..='z' {
        let match_string = format!("{0}{0}", character);

        if input.contains(&match_string)
        {
            return true;
        }
    }
```

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

# Improvements

Could be cleaner, like that I did most of it functionally.