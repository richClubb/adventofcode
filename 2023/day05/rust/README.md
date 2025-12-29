Advent of code 2023 Day 05 - Rust
---------------------------------

# Overview

This was a lot easier than I'd originally expected and was a lot of fun.

I came into it trying to be as idiomatic as possible, trying to minimise mutability and use the more functional approach. The overflow checks were very helpful in the testing as they recognised when I'd overflowed a variable with the wrong maths.

The use of types like `Result` are just pleasant, as well as the match syntax just makes code nice to read.

I think the main draw was the parallelisation. Because I'd been functional, I was able to change just one function call between the single threaded and parallel versions

```Rust
pub fn part_b_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000000000);
    let map_layers = get_map_layers_from_file(&path);

    let min_value = seed_ranges.iter().map(|a| a.get_lowest_seed_in_range(&map_layers)).min().unwrap();

    println!("Part B forward brute force: {min_value}");
}

pub fn part_b_parallel_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000);
    let map_layers = get_map_layers_from_file(&path);

    let result = seed_ranges.par_iter().map(|s| s.get_lowest_seed_in_range(&map_layers)).min().unwrap();

    println!("Part B forward parallel: {result}");

}
```

Going from `iter()` to `par_iter()` was just fantastic.

# Build / Run

```
cargo build
cargo run [path] [run]
```

Where run is: 
* part_a
* part_b_forward
* part_b_inverse
* part_b_parallel_forward
* part_b_parallel_inverse
* part_b_ranges (not implemented);

It builds debug by default but you can specify release by either

```
cargo run --release [path] [run]
```

or
```
cargo build --release
./target/release/day5 [path] [run]
```

# Test

```
cargo test
```