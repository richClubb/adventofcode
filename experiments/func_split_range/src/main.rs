
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

pub fn split_range(input_range: SeedRange, new_max_size: u64) -> Vec<SeedRange> {

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
        let start_2: u64 = input_range.start + new_max_size - 1;
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
                SeedRange::new(input_range.start - 1 + new_max_size, input_range.size - new_max_size), 
                new_max_size
            )
            ].concat()
}


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_map_1() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(test_map, 10);

        assert_eq!(new_ranges.len(), 1);
        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].size, 10);

    }

    #[test]
    fn split_map_2() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(test_map, 9);

        assert_eq!(new_ranges.len(), 2);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].size,  9);
        assert_eq!(new_ranges[1].start, 9);
        assert_eq!(new_ranges[1].size,  1);
    }

    #[test]
    fn split_map_3() {
        let test_map: SeedRange = SeedRange::new(1, 10);

        let new_ranges = split_range(test_map, 4);

        assert_eq!(new_ranges.len(), 3);

        assert_eq!(new_ranges[0].start, 1);
        assert_eq!(new_ranges[0].size,  4);
        assert_eq!(new_ranges[1].start, 4);
        assert_eq!(new_ranges[1].size,  4);
        assert_eq!(new_ranges[2].start, 8);
        assert_eq!(new_ranges[2].size,  2);
    }
}