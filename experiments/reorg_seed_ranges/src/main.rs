
mod seed_range;

fn main() {
    println!("Hello, world!");

    let mut test_seed_ranges = seed_range::SeedRanges{seed_ranges: Vec::new()};

    test_seed_ranges.add_seed_range(seed_range::SeedRange::new(1, 5));
    test_seed_ranges.add_seed_range(seed_range::SeedRange::new(10, 5));
    test_seed_ranges.add_seed_range(seed_range::SeedRange::new(15, 5));
    test_seed_ranges.add_seed_range(seed_range::SeedRange::new(20, 5));

    test_seed_ranges.split_ranges(20);
}


