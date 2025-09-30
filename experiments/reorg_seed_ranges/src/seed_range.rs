// pub mod seed_range;

pub struct SeedRange{
    pub start: u64,
    pub end: u64,
    pub size: u64
}

impl SeedRange{
    pub fn new(start: u64, size: u64) -> SeedRange {
        return SeedRange{ 
            start: start,
            end: start + size,
            size: size,
        };
    }

    pub fn split_range(&self, new_size: u64) -> Vec<SeedRange> {

        if (new_size >= self.size)
        {
            return vec![SeedRange::new(self.start, self.size)];
        }
        else if (new_size == self.size / 2)
        {
            return vec![
                SeedRange::new(self.start, new_size),
                SeedRange::new(self.start + new_size, self.size - new_size)
            ];
        }
        else if ( 
            (self.size / new_size == 1) && 
            (self.size % new_size != 0 )
        )
        {
            let size_1 = self.size / 2;
            let size_2 = self.size - size_1;
            return vec![
                SeedRange::new(self.start, size_1), 
                SeedRange::new(self.start + size_1, size_2)    
            ];
        } 

        let mut new_seed_ranges: Vec<SeedRange> = Vec::new();
        let mut new_start = self.start;
        let mut remaining = self.size;

        while(remaining != 0)
        {
            if (remaining >= new_size)
            {
                new_seed_ranges.push(SeedRange::new(new_start, new_size));
                remaining -= new_size;
                new_start += new_size;
            }
            else {
                new_seed_ranges.push(SeedRange::new(new_start, remaining));
                remaining = 0;
            }
        }

        return new_seed_ranges;
    }
}

pub struct SeedRanges{
    pub seed_ranges: Vec<SeedRange>,
}

impl SeedRanges{

    pub fn add_seed_range(&mut self, seed_range: SeedRange) {
        self.seed_ranges.push(seed_range);
    }

    pub fn split_ranges(&mut self, max_size: u64) {
        println!("Size {}", self.seed_ranges.len());
        let total = self.seed_ranges.iter().fold(0, |mut total, seed_range| {total += seed_range.size; total});

        let ideal_size = total / max_size;
        let ideal_size_remainder = total % max_size;

        if ((ideal_size_remainder == 0) && (total == max_size))
        {
            println!("Size-of 1");
        }
        else if (ideal_size_remainder == 0)
        {
            println!("no remainder");
        }
        else 
        {
            println!("Remainder");
        }  
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_seed_range() {

        let test_seed_range = SeedRange::new(1, 5);
        let new_seed_ranges = test_seed_range.split_range(5);
        assert_eq!(new_seed_ranges.len(), 1);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 5);
        assert_eq!(new_seed_ranges[0].end, 6);

        let test_seed_range = SeedRange::new(1, 5);
        let new_seed_ranges = test_seed_range.split_range(4);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 2);
        assert_eq!(new_seed_ranges[0].end, 3);
        assert_eq!(new_seed_ranges[1].start, 3);
        assert_eq!(new_seed_ranges[1].size, 3);
        assert_eq!(new_seed_ranges[1].end, 6);

        let test_seed_range = SeedRange::new(1, 5);
        let new_seed_ranges = test_seed_range.split_range(6);
        assert_eq!(new_seed_ranges.len(), 1);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 5);
        assert_eq!(new_seed_ranges[0].end, 6);

        let test_seed_range = SeedRange::new(1, 4);
        let new_seed_ranges = test_seed_range.split_range(2);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 2);
        assert_eq!(new_seed_ranges[0].end, 3);
        assert_eq!(new_seed_ranges[1].start, 3);
        assert_eq!(new_seed_ranges[1].size, 2);
        assert_eq!(new_seed_ranges[1].end, 5);

        let test_seed_range = SeedRange::new(1, 4);
        let new_seed_ranges = test_seed_range.split_range(3);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 2);
        assert_eq!(new_seed_ranges[0].end, 3);
        assert_eq!(new_seed_ranges[1].start, 3);
        assert_eq!(new_seed_ranges[1].size, 2);
        assert_eq!(new_seed_ranges[1].end, 5);

        let test_seed_range = SeedRange::new(1, 5);
        let new_seed_ranges = test_seed_range.split_range(2);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 2);
        assert_eq!(new_seed_ranges[0].end, 3);
        assert_eq!(new_seed_ranges[1].start, 3);
        assert_eq!(new_seed_ranges[1].size, 3);
        assert_eq!(new_seed_ranges[1].end, 6);

        let test_seed_range = SeedRange::new(1, 5);
        let new_seed_ranges = test_seed_range.split_range(2);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 1);
        assert_eq!(new_seed_ranges[0].size, 2);
        assert_eq!(new_seed_ranges[0].end, 3);
        assert_eq!(new_seed_ranges[1].start, 3);
        assert_eq!(new_seed_ranges[1].size, 3);
        assert_eq!(new_seed_ranges[1].end, 6);

        let test_seed_range = SeedRange::new(2150339939, 385349830);
        let new_seed_ranges = test_seed_range.split_range(192674915);
        assert_eq!(new_seed_ranges.len(), 2);
        assert_eq!(new_seed_ranges[0].start, 2150339939);
        assert_eq!(new_seed_ranges[0].size, 192674915);
        assert_eq!(new_seed_ranges[0].end, 2343014854);
        assert_eq!(new_seed_ranges[1].start, 2343014854);
        assert_eq!(new_seed_ranges[1].size, 192674915);
        assert_eq!(new_seed_ranges[1].end, 2535689769);

        let test_seed_range = SeedRange::new(2150339939, 385349830);
        let new_seed_ranges = test_seed_range.split_range(128449944);
        assert_eq!(new_seed_ranges.len(), 3);
        assert_eq!(new_seed_ranges[0].start, 2150339939);
        assert_eq!(new_seed_ranges[0].size, 128449944);
        assert_eq!(new_seed_ranges[0].end, 2278789883);

        assert_eq!(new_seed_ranges[1].start, 2278789883);
        assert_eq!(new_seed_ranges[1].size, 128449944);
        assert_eq!(new_seed_ranges[1].end, 2407239827);

        assert_eq!(new_seed_ranges[2].start, 2407239827);
        assert_eq!(new_seed_ranges[2].size, 128449942);
        assert_eq!(new_seed_ranges[2].end, 2535689769);
    }

    #[test]
    fn add_seed_range() {

        let mut test_seed_ranges = SeedRanges{seed_ranges: Vec::new()};

        test_seed_ranges.add_seed_range(SeedRange::new(1, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(10, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(15, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(20, 5));

        assert_eq!(test_seed_ranges.seed_ranges.len(), 4);
    }

    #[test]
    fn split_seed_ranges() {

        let mut test_seed_ranges = SeedRanges{seed_ranges: Vec::new()};

        test_seed_ranges.add_seed_range(SeedRange::new(1, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(10, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(15, 5));
        test_seed_ranges.add_seed_range(SeedRange::new(20, 5));

        test_seed_ranges.split_ranges(20);

        assert_eq!(test_seed_ranges.seed_ranges.len(), 4);
    }

}