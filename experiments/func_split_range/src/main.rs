
#[derive(Clone)]
pub struct SeedRange{
    pub start: u64,
    pub end: u64,
    pub size: u64
}

impl SeedRange{

    pub fn new(start: u64, size: u64) -> SeedRange {
        return SeedRange{start: start, end: start + size, size: size}
    }
}

pub fn split_range(input_range: &SeedRange, new_max_size: u64) -> Vec<SeedRange> {

    let ideal_size = find_ideal_size(&input_range.size, &new_max_size);
    println!("Ideal size: {} {}", ideal_size, new_max_size);

    if input_range.size <= ideal_size {
        println!("small or equal {}, {}", input_range.start, input_range.size);
        return vec![
            SeedRange::new(input_range.start, input_range.size)
        ]
    }

    if (input_range.size / ideal_size) == 1 {
        
        let start_1: u64 = input_range.start;
        let size_1: u64 = ideal_size;
        let start_2: u64 = input_range.start + ideal_size;
        let size_2: u64 = input_range.size - ideal_size;

        println!("even split {}, {}, {}, {}", start_1, size_1, start_2, size_2);
        return vec![
            SeedRange::new(start_1, size_1),
            SeedRange::new(start_2, size_2)
        ]
    }

    println!("recurse {}, {}, {}, {}", input_range.start, ideal_size, input_range.start + ideal_size, input_range.size - ideal_size);
    return vec![ 
        vec![
            SeedRange::new(input_range.start, ideal_size)], 
            split_range(
                &SeedRange::new(input_range.start + ideal_size, input_range.size - ideal_size), 
                ideal_size
            )
            ].concat()
}

pub fn find_ideal_size(range_len: &u64, new_max_size: &u64) -> u64 {

    println!("find ideal {}, {}", range_len, new_max_size);
    // array two small - return
    if new_max_size > range_len {
        println!("Bigger");
        return *new_max_size;
    }

    let starting_value = range_len / new_max_size;
    let starting_value_rem = range_len % new_max_size;

    println!("starting val {}.{}", starting_value, starting_value_rem);
    // even division - return
    if starting_value_rem == 0 {
        return *new_max_size;
    }

    let next_new_max_size = new_max_size - 1;
    let next_starting_value = (range_len / next_new_max_size);
    let next_starting_value_rem = (range_len % next_new_max_size);
    println!("next starting val {}.{}", next_starting_value, next_starting_value_rem);

    // tries to find where the boundary is before it goes into the next integer
    if (next_starting_value > starting_value)
    {
        if next_starting_value >= (starting_value + 1)
        {
            return starting_value;
        }

        if next_new_max_size == 1 {
            return *new_max_size;
        }

        if range_len % next_new_max_size == 0 {
            return *new_max_size - 1
        }

        return *new_max_size;
    }

    return find_ideal_size(range_len, &next_new_max_size)
}


pub fn print_seed_range(seed_range: &SeedRange) {
    println!("start: {} end: {} size: {}", seed_range.start, seed_range.end, seed_range.size);
}

pub fn split_ranges(seed_ranges: &Vec<SeedRange>, new_max_size: u64) -> Vec<SeedRange> {

    let new_ranges: Vec<SeedRange> = seed_ranges
        .into_iter()
        .fold(
            Vec::<SeedRange>::new(),
            |acc, seed_range| acc.into_iter().chain(split_range(&seed_range, new_max_size)).collect()
        );
    
    return new_ranges;

}

pub fn split_ranges_by_num(seed_ranges: &Vec<SeedRange>, new_max_num_ranges: u64) -> Vec<SeedRange> {

    if seed_ranges.len() < new_max_num_ranges.try_into().unwrap() {
        return seed_ranges.clone();
    }

    let total: u64 = seed_ranges.into_iter().fold(0, |acc, seed_range| acc + seed_range.size);

    let num_ranges_rem = total % new_max_num_ranges;
    
    let num_ranges = if num_ranges_rem == 0 {
        total / new_max_num_ranges
    }
    else {
        (total / new_max_num_ranges) + 1
    };

    let new_max_range_size = if total % num_ranges == 0 {
        total / num_ranges
    } else {
        (total / num_ranges) + 1
    };

    return split_ranges(seed_ranges, new_max_range_size);

}

fn main() {
    println!("Hello, world!");

    // let seed_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 2), SeedRange::new(3, 4)];

    // split_ranges(&seed_ranges, 10);
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn split_range_1() {
    //     let test_map: SeedRange = SeedRange::new(1, 10);

    //     let new_ranges = split_range(&test_map, 10);

    //     assert_eq!(new_ranges.len(), 1);
    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].end,  11);
    //     assert_eq!(new_ranges[0].size, 10);

    // }

    // #[test]
    // fn split_range_2() {
    //     let test_map: SeedRange = SeedRange::new(1, 10);

    //     let new_ranges = split_range(&test_map, 9);

    //     assert_eq!(new_ranges.len(), 2);

    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].size,  5);
    //     assert_eq!(new_ranges[0].end,   6);
    //     assert_eq!(new_ranges[1].start, 6);
    //     assert_eq!(new_ranges[1].size,  5);
    //     assert_eq!(new_ranges[1].end,   11);
    // }

    // #[test]
    // fn split_range_3() {
    //     let test_map: SeedRange = SeedRange::new(1, 10);

    //     let new_ranges = split_range(&test_map, 4);

    //     assert_eq!(new_ranges.len(), 3);

    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].end,   5);
    //     assert_eq!(new_ranges[0].size,  4);

    //     assert_eq!(new_ranges[1].start, 5);
    //     assert_eq!(new_ranges[1].end,   8);
    //     assert_eq!(new_ranges[1].size,  3);
        
    //     assert_eq!(new_ranges[2].start, 8);
    //     assert_eq!(new_ranges[2].end,  11);
    //     assert_eq!(new_ranges[2].size,  3);
    // }

    // #[test]
    // fn split_range_4() {
    //     let test_map: SeedRange = SeedRange::new(1, 10);

    //     let new_ranges = split_range(&test_map, 3);

    //     assert_eq!(new_ranges.len(), 4);

    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].end,   4);
    //     assert_eq!(new_ranges[0].size,  3);

    //     assert_eq!(new_ranges[1].start, 4);
    //     assert_eq!(new_ranges[1].end,   7);
    //     assert_eq!(new_ranges[1].size,  3);
        
    //     assert_eq!(new_ranges[2].start, 7);
    //     assert_eq!(new_ranges[2].end,   9);
    //     assert_eq!(new_ranges[2].size,  2);

    //     assert_eq!(new_ranges[3].start,  9);
    //     assert_eq!(new_ranges[3].end,   11);
    //     assert_eq!(new_ranges[3].size,   2);
    // }

    // #[test]
    // fn split_ranges_1() {
    //     let test_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 10), SeedRange::new(20,10)];

    //     let new_ranges = split_ranges(&test_ranges, 5);

    //     assert_eq!(new_ranges.len(), 4);

    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].end,   6);
    //     assert_eq!(new_ranges[0].size,  5);

    //     assert_eq!(new_ranges[1].start, 6);
    //     assert_eq!(new_ranges[1].end,   11);
    //     assert_eq!(new_ranges[1].size,  5);
        
    //     assert_eq!(new_ranges[2].start, 20);
    //     assert_eq!(new_ranges[2].end,   25);
    //     assert_eq!(new_ranges[2].size,   5);

    //     assert_eq!(new_ranges[3].start, 25);
    //     assert_eq!(new_ranges[3].end,   30);
    //     assert_eq!(new_ranges[3].size,   5);
    // }

    // #[test]
    // fn split_ranges_2() {
    //     let test_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 10), SeedRange::new(20,6)];

    //     let new_ranges = split_ranges(&test_ranges, 3);

    //     assert_eq!(new_ranges.len(), 6);

    //     assert_eq!(new_ranges[0].start, 1);
    //     assert_eq!(new_ranges[0].end,   4);
    //     assert_eq!(new_ranges[0].size,  3);

    //     assert_eq!(new_ranges[1].start, 4);
    //     assert_eq!(new_ranges[1].end,   7);
    //     assert_eq!(new_ranges[1].size,  3);
        
    //     assert_eq!(new_ranges[2].start, 7);
    //     assert_eq!(new_ranges[2].end,   10);
    //     assert_eq!(new_ranges[2].size,   3);

    //     assert_eq!(new_ranges[3].start, 10);
    //     assert_eq!(new_ranges[3].end,   11);
    //     assert_eq!(new_ranges[3].size,   1);

    //     assert_eq!(new_ranges[4].start, 20);
    //     assert_eq!(new_ranges[4].end,   23);
    //     assert_eq!(new_ranges[4].size,   3);

    //     assert_eq!(new_ranges[5].start, 23);
    //     assert_eq!(new_ranges[5].end,   26);
    //     assert_eq!(new_ranges[5].size,   3);
    // }

    #[test]
    fn find_ideal_size_test() {

        let new_value = find_ideal_size(&19, &50);
        assert_eq!(new_value, 50);

        let new_value = find_ideal_size(&19, &20);
        assert_eq!(new_value, 20);

        let new_value = find_ideal_size(&19, &19);
        assert_eq!(new_value, 19);

        let new_value = find_ideal_size(&19, &18);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &17);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &16);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &15);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &14);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &13);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &12);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &11);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &10);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&19, &9);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&19, &8);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&19, &7);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&19, &6);
        assert_eq!(new_value, 5);

        let new_value = find_ideal_size(&19, &5);
        assert_eq!(new_value, 5);

        let new_value = find_ideal_size(&19, &4);
        assert_eq!(new_value, 4);

        let new_value = find_ideal_size(&19, &3);
        assert_eq!(new_value, 3);

        let new_value = find_ideal_size(&19, &2);
        assert_eq!(new_value, 2);

        let new_value = find_ideal_size(&19, &1);
        assert_eq!(new_value, 1);

        let new_value = find_ideal_size(&10, &9);
        assert_eq!(new_value, 5);

        let new_value = find_ideal_size(&4, &3);
        assert_eq!(new_value, 2);
    }
}