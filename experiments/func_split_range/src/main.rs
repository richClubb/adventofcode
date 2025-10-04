
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

    if input_range.size <= new_max_size {
        println!("small or equal");
        return vec![
            SeedRange::new(input_range.start, input_range.size)
        ]
    }

    if (input_range.size / new_max_size) == 1 {
        println!("even split");
        let start_1: u64 = input_range.start;
        let size_1: u64 = new_max_size;
        let start_2: u64 = input_range.start + new_max_size;
        let size_2: u64 = input_range.size - new_max_size;

        return vec![
            SeedRange::new(start_1, size_1),
            SeedRange::new(start_2, size_2)
        ]
        
    }

    return vec![ 
        vec![
            SeedRange::new(input_range.start, new_max_size)], 
            split_range(
                &SeedRange::new(input_range.start + new_max_size, input_range.size - new_max_size), 
                new_max_size
            )
            ].concat()
}

pub fn print_seed_range(seed_range: &SeedRange) {
    println!("start: {} end: {} size: {}", seed_range.start, seed_range.end, seed_range.size);
}

pub fn split_ranges(seed_ranges: &Vec<SeedRange>, new_max_size: u64) -> Vec<SeedRange> {
    
    // seed_ranges.iter().for_each(|seed_range| print_seed_range(seed_range));
    let new_ranges: Vec<SeedRange> = seed_ranges
        .into_iter()
        .fold(
            Vec::<SeedRange>::new(),
            |acc, seed_range| acc.into_iter().chain(split_range(&seed_range, new_max_size)).collect()
        );
    
    return new_ranges;

}

fn main() {
    println!("Hello, world!");

    let seed_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 2), SeedRange::new(3, 4)];

    split_ranges(&seed_ranges, 10);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_ranges_1() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 10);

        assert_eq!(new_ranges.len(), 1);
        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,  11);
        assert_eq!(new_ranges[0].size, 10);

    }

    #[test]
    fn split_ranges_2() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 9);

        assert_eq!(new_ranges.len(), 2);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].size,  9);
        assert_eq!(new_ranges[0].end,  10);
        assert_eq!(new_ranges[1].start,10);
        assert_eq!(new_ranges[1].size,  1);
    }

    #[test]
    fn split_ranges_3() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(&test_map, 4);

        assert_eq!(new_ranges.len(), 3);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   5);
        assert_eq!(new_ranges[0].size,  4);

        assert_eq!(new_ranges[1].start, 5);
        assert_eq!(new_ranges[1].end,   9);
        assert_eq!(new_ranges[1].size,  4);
        
        assert_eq!(new_ranges[2].start, 9);
        assert_eq!(new_ranges[2].end,  11);
        assert_eq!(new_ranges[2].size,  2);
    }

    #[test]
    fn split_ranges_4() {
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
        assert_eq!(new_ranges[2].end,  10);
        assert_eq!(new_ranges[2].size,  3);

        assert_eq!(new_ranges[3].start, 10);
        assert_eq!(new_ranges[3].end,   11);
        assert_eq!(new_ranges[3].size,   1);
    }

    #[test]
    fn split_ranges_5() {
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
    fn split_ranges_6() {
        let test_ranges: Vec<SeedRange> = vec![SeedRange::new(1, 10), SeedRange::new(20,6)];

        let new_ranges = split_ranges(&test_ranges, 3);

        assert_eq!(new_ranges.len(), 6);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].end,   4);
        assert_eq!(new_ranges[0].size,  3);

        assert_eq!(new_ranges[1].start, 4);
        assert_eq!(new_ranges[1].end,   7);
        assert_eq!(new_ranges[1].size,  3);
        
        assert_eq!(new_ranges[2].start, 7);
        assert_eq!(new_ranges[2].end,   10);
        assert_eq!(new_ranges[2].size,   3);

        assert_eq!(new_ranges[3].start, 10);
        assert_eq!(new_ranges[3].end,   11);
        assert_eq!(new_ranges[3].size,   1);

        assert_eq!(new_ranges[4].start, 20);
        assert_eq!(new_ranges[4].end,   23);
        assert_eq!(new_ranges[4].size,   3);

        assert_eq!(new_ranges[5].start, 23);
        assert_eq!(new_ranges[5].end,   26);
        assert_eq!(new_ranges[5].size,   3);
    }
}