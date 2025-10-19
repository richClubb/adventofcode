
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

    if input_range.size <= ideal_size {
        return vec![
            SeedRange::new(input_range.start, input_range.size)
        ]
    }

    if (input_range.size / ideal_size) == 1 {
        
        let start_1: u64 = input_range.start;
        let size_1: u64 = ideal_size;
        let start_2: u64 = input_range.start + ideal_size;
        let size_2: u64 = input_range.size - ideal_size;

        return vec![
            SeedRange::new(start_1, size_1),
            SeedRange::new(start_2, size_2)
        ]
    }

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

    if *new_max_size == 1 {
        println!("a");
        return 1;
    }

    if *range_len <= *new_max_size {
        println!("b");
        return *range_len;
    }

    let initial_attempt = range_len / new_max_size;
    let initial_attempt_rem = range_len % new_max_size;

    if initial_attempt_rem == 0 {
        println!("c");
        return *new_max_size;
    }

    if (initial_attempt == 2) && (initial_attempt_rem == 0)
    {
        println!("d");
        return *range_len / 2;
    }

    let divisor = initial_attempt + 1;
    let new_size_initial = range_len / divisor;
    let new_size_initial_rem = range_len % divisor;

    if new_size_initial_rem == 0
    {
        println!("e");
        return new_size_initial;
    }
    
    println!("f");
    return new_size_initial + 1;
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

    if seed_ranges.len() as u64 >= new_max_num_ranges {
        return seed_ranges.clone();
    }

    let total: u64 = seed_ranges.into_iter().fold(0, |acc, seed_range| acc + seed_range.size);

    let ideal_range_size_rem = total % new_max_num_ranges;

    let ideal_range_size = if ideal_range_size_rem == 0 {
        total / new_max_num_ranges
    }
    else {
        (total / new_max_num_ranges) + 1
    };

    if seed_ranges.len() == 1 {
        return split_range(&seed_ranges[0], ideal_range_size);
    }

    let base_ranges = split_range(&seed_ranges[0], ideal_range_size);
    let next_ranges = split_ranges_by_num(&seed_ranges[1..].to_vec(), new_max_num_ranges - base_ranges.len() as u64);
    return vec![base_ranges, next_ranges].concat()
}

fn main() {
    println!("Hello, world!");

    let test_ranges = vec![SeedRange::new(1, 10), SeedRange::new(21,10)];

    let result = split_ranges_by_num(&test_ranges, 4);
    println!("result {}", result.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_range_1() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 10);

        assert_eq!(new_ranges.len(), 1);
        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,  11);
        assert_eq!(new_ranges[0].size, 10);

    }

    #[test]
    fn split_range_2() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 9);

        assert_eq!(new_ranges.len(), 2);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].size,  5);
        assert_eq!(new_ranges[0].end,   6);
        assert_eq!(new_ranges[1].start, 6);
        assert_eq!(new_ranges[1].size,  5);
        assert_eq!(new_ranges[1].end,   11);
    }

    #[test]
    fn split_range_3() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 4);

        assert_eq!(new_ranges.len(), 3);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   5);
        assert_eq!(new_ranges[0].size,  4);

        assert_eq!(new_ranges[1].start, 5);
        assert_eq!(new_ranges[1].end,   8);
        assert_eq!(new_ranges[1].size,  3);
        
        assert_eq!(new_ranges[2].start, 8);
        assert_eq!(new_ranges[2].end,  11);
        assert_eq!(new_ranges[2].size,  3);

        let new_ranges = split_range(&test_map, 5);

        assert_eq!(new_ranges.len(), 2);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   6);
        assert_eq!(new_ranges[0].size,  5);

        assert_eq!(new_ranges[1].start, 6);
        assert_eq!(new_ranges[1].end,   11);
        assert_eq!(new_ranges[1].size,  5);
    }

    #[test]
    fn split_range_4() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 3);

        assert_eq!(new_ranges.len(), 4);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   4);
        assert_eq!(new_ranges[0].size,  3);

        assert_eq!(new_ranges[1].start, 4);
        assert_eq!(new_ranges[1].end,   7);
        assert_eq!(new_ranges[1].size,  3);
        
        assert_eq!(new_ranges[2].start, 7);
        assert_eq!(new_ranges[2].end,   9);
        assert_eq!(new_ranges[2].size,  2);

        assert_eq!(new_ranges[3].start,  9);
        assert_eq!(new_ranges[3].end,   11);
        assert_eq!(new_ranges[3].size,   2);
    }

    #[test]
    fn split_range_5() {
        let test_map: SeedRange = SeedRange::new(28965817, 302170009);

        let new_ranges = split_range(&test_map, 100000000);

        assert_eq!(new_ranges.len(), 4);
    }

    #[test]
    fn split_ranges_1() {
        let test_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 10), SeedRange::new(20,10)];

        let new_ranges = split_ranges(&test_ranges, 5);

        assert_eq!(new_ranges.len(), 4);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   6);
        assert_eq!(new_ranges[0].size,  5);

        assert_eq!(new_ranges[1].start, 6);
        assert_eq!(new_ranges[1].end,   11);
        assert_eq!(new_ranges[1].size,  5);
        
        assert_eq!(new_ranges[2].start, 20);
        assert_eq!(new_ranges[2].end,   25);
        assert_eq!(new_ranges[2].size,   5);

        assert_eq!(new_ranges[3].start, 25);
        assert_eq!(new_ranges[3].end,   30);
        assert_eq!(new_ranges[3].size,   5);
    }

    #[test]
    fn split_ranges_2() {
        let test_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 10), SeedRange::new(20,6)];

        let new_ranges = split_ranges(&test_ranges, 3);

        assert_eq!(new_ranges.len(), 6);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   4);
        assert_eq!(new_ranges[0].size,  3);

        assert_eq!(new_ranges[1].start, 4);
        assert_eq!(new_ranges[1].end,   7);
        assert_eq!(new_ranges[1].size,  3);
        
        assert_eq!(new_ranges[2].start,  7);
        assert_eq!(new_ranges[2].end,    9);
        assert_eq!(new_ranges[2].size,   2);

        assert_eq!(new_ranges[3].start,  9);
        assert_eq!(new_ranges[3].end,   11);
        assert_eq!(new_ranges[3].size,   2);

        assert_eq!(new_ranges[4].start, 20);
        assert_eq!(new_ranges[4].end,   23);
        assert_eq!(new_ranges[4].size,   3);

        assert_eq!(new_ranges[5].start, 23);
        assert_eq!(new_ranges[5].end,   26);
        assert_eq!(new_ranges[5].size,   3);
    }

    #[test]
    fn find_ideal_size_oversize_test() {

        let new_value = find_ideal_size(&19, &50);
        assert_eq!(new_value, 19);

        let new_value = find_ideal_size(&19, &20);
        assert_eq!(new_value, 19);
    }

    #[test]
    fn find_ideal_size_equal_test() {

        let new_value = find_ideal_size(&19, &19);
        assert_eq!(new_value, 19);

        let new_value = find_ideal_size(&20, &20);
        assert_eq!(new_value, 20);

        let new_value = find_ideal_size(&1, &1);
        assert_eq!(new_value, 1);

        let new_value = find_ideal_size(&15, &15);
        assert_eq!(new_value, 15);
    }

    #[test]
    fn find_ideal_size_odd() {
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
    }

    #[test]
    fn find_ideal_size_even() {
        let new_value = find_ideal_size(&20, &20);
        assert_eq!(new_value, 20);

        let new_value = find_ideal_size(&20, &19);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &18);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &17);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &16);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &15);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &14);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &13);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &12);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &11);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &10);
        assert_eq!(new_value, 10);

        let new_value = find_ideal_size(&20, &9);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&20, &8);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&20, &7);
        assert_eq!(new_value, 7);

        let new_value = find_ideal_size(&20, &6);
        assert_eq!(new_value, 5);

        let new_value = find_ideal_size(&20, &5);
        assert_eq!(new_value, 5);

        let new_value = find_ideal_size(&20, &4);
        assert_eq!(new_value, 4);

        let new_value = find_ideal_size(&20, &3);
        assert_eq!(new_value, 3);

        let new_value = find_ideal_size(&20, &2);
        assert_eq!(new_value, 2);

        let new_value = find_ideal_size(&20, &1);
        assert_eq!(new_value, 1);
    }

    #[test]
    fn find_ideal_size_large_numbers() {
        assert_eq!(find_ideal_size(&302170009, &100000000), 75542503);

        assert_eq!(find_ideal_size(&48290258, &10000000), 9658052);

        assert_eq!(find_ideal_size(&243492043, &100000000), 81164015);

        assert_eq!(find_ideal_size(&385349830, &100000000), 96337458);

        assert_eq!(find_ideal_size(&350474859, &100000000), 87618715);

        assert_eq!(find_ideal_size(&17565716, &10000000), 8782858);

        assert_eq!(find_ideal_size(&291402104, &100000000), 97134035);

        assert_eq!(find_ideal_size(&279196488, &100000000), 93065496);

        assert_eq!(find_ideal_size(&47952959, &10000000), 9590592);

        assert_eq!(find_ideal_size(&9607836, &1000000), 960784);

        assert_eq!(find_ideal_size(&302170009, &10000000), 9747420);

        assert_eq!(find_ideal_size(&302170009, &1000000), 997261);

        assert_eq!(find_ideal_size(&302170009, &100000), 99991);

        assert_eq!(find_ideal_size(&302170009, &10000), 10000);

        assert_eq!(find_ideal_size(&302170009, &1000), 1000);

        assert_eq!(find_ideal_size(&302170009, &100), 100);

        assert_eq!(find_ideal_size(&302170009, &10), 10);

        assert_eq!(find_ideal_size(&302170009, &1), 1);
    }

    #[test]
    fn split_ranges_by_num_test() {
        
        let test_ranges = vec![SeedRange::new(1, 10), SeedRange::new(21,10)];

        let result = split_ranges_by_num(&test_ranges, 2);
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].start,   1);
        assert_eq!(result[0].end,    11);
        assert_eq!(result[0].size,   10);

        assert_eq!(result[1].start,  21);
        assert_eq!(result[1].end,    31);
        assert_eq!(result[1].size,   10);

        let result = split_ranges_by_num(&test_ranges, 1);
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].start,   1);
        assert_eq!(result[0].end,    11);
        assert_eq!(result[0].size,   10);

        assert_eq!(result[1].start,  21);
        assert_eq!(result[1].end,    31);
        assert_eq!(result[1].size,   10);

        let result = split_ranges_by_num(&test_ranges, 4);
        assert_eq!(result.len(), 4);

        assert_eq!(result[0].start,  1);
        assert_eq!(result[0].end,    6);
        assert_eq!(result[0].size,   5);

        assert_eq!(result[1].start,  6);
        assert_eq!(result[1].end,   11);
        assert_eq!(result[1].size,   5);
    
        assert_eq!(result[2].start, 21);
        assert_eq!(result[2].end,   26);
        assert_eq!(result[2].size,   5);

        assert_eq!(result[3].start, 26);
        assert_eq!(result[3].end,   31);
        assert_eq!(result[3].size,   5);

        let result = split_ranges_by_num(&test_ranges, 3);
        assert_eq!(result.len(), 3);

        assert_eq!(result[0].start,  1);
        assert_eq!(result[0].end,    6);
        assert_eq!(result[0].size,   5);

        assert_eq!(result[1].start,  6);
        assert_eq!(result[1].end,   11);
        assert_eq!(result[1].size,   5);
    
        assert_eq!(result[2].start, 21);
        assert_eq!(result[2].end,   31);
        assert_eq!(result[2].size,  10);

        let result = split_ranges_by_num(&test_ranges, 7);
        assert_eq!(result.len(), 7);

        assert_eq!(result[0].start,  1);
        assert_eq!(result[0].end,    4);
        assert_eq!(result[0].size,   3);

        assert_eq!(result[1].start,  4);
        assert_eq!(result[1].end,    7);
        assert_eq!(result[1].size,   3);
    
        assert_eq!(result[2].start,  7);
        assert_eq!(result[2].end,    9);
        assert_eq!(result[2].size,   2);

        assert_eq!(result[3].start,  9);
        assert_eq!(result[3].end,   11);
        assert_eq!(result[3].size,   2);

        assert_eq!(result[4].start, 21);
        assert_eq!(result[4].end,   25);
        assert_eq!(result[4].size,   4);

        assert_eq!(result[5].start, 25);
        assert_eq!(result[5].end,   28);
        assert_eq!(result[5].size,   3);

        assert_eq!(result[6].start, 28);
        assert_eq!(result[6].end,   31);
        assert_eq!(result[6].size,   3);

    }

    #[test]
    fn split_range_by_num_large_ranges() {
        
        let test_ranges = vec![
            SeedRange::new( 1828835733,   9607836),
            SeedRange::new( 2566296746,  17565716),
            SeedRange::new( 3227221259,  47952959),
            SeedRange::new( 1752849261,  48290258),
            SeedRange::new(  804904201, 243492043),
            SeedRange::new(  447111316, 279196488),
            SeedRange::new( 3543571814, 291402104),
            SeedRange::new(   28965817, 302170009), 
            SeedRange::new( 1267802202, 350474859),
            SeedRange::new( 2150339939, 385349830),
        ];

        let result = split_ranges_by_num(&test_ranges, 11);
        assert!(result.len() <= 3000);
    }

    #[test]
    fn split_ranges_by_num_sample_data_test() {
        
        let test_ranges = vec![
            SeedRange::new( 1828835733,   9607836),
            SeedRange::new( 2566296746,  17565716),
            SeedRange::new( 3227221259,  47952959),
            SeedRange::new( 1752849261,  48290258),
            SeedRange::new(  804904201, 243492043),
            SeedRange::new(  447111316, 279196488),
            SeedRange::new( 3543571814, 291402104),
            SeedRange::new(   28965817, 302170009), 
            SeedRange::new( 1267802202, 350474859),
            SeedRange::new( 2150339939, 385349830),
        ];

        let result = split_ranges_by_num(&test_ranges, 11);
        assert!(result.len() <= 3000);
    }
}