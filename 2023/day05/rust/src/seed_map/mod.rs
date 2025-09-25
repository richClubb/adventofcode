use crate::seed::Seed;
use crate::seed_range::SeedRange;

pub struct SeedMap {
    pub dest_start: u64,
    pub dest_end: u64,
    pub src_start: u64,
    pub src_end: u64,
    pub size: u64,
}

impl SeedMap {

    pub fn new(source: u64, dest: u64, size: u64) -> SeedMap {
        return SeedMap{ 
            src_start: source, 
            src_end: source + size, 
            dest_start: dest,
            dest_end: dest + size,
            size: size
        };
    }

    pub fn map_seed(&self, seed: &Seed) -> Option<Seed>
    {
        if ( seed.value < self.src_start ) || ( seed.value >= self.src_end )
        {
            return None;
        }
        
        return Some(Seed{value: (seed.value - self.src_start) + self.dest_start});
    }

    pub unsafe fn map_seed_ptr(&self, seed_value: *mut u64) -> bool 
    {
        if( *seed_value < self.src_start ) || ( *seed_value >= self.src_end )
        {
            return false;
        }

        *seed_value = (*seed_value - self.src_start) + self.dest_start;
        return true;
    }

    pub fn map_seed_range(&self, seed_range: &SeedRange) -> (Option<Vec<SeedRange>>, Option<SeedRange>)
    {
        return self.case_1(seed_range);
    }

    fn case_1(&self, seed_range: &SeedRange) -> (Option<Vec<SeedRange>>, Option<SeedRange>)
    {
        if seed_range.end < self.src_start
        {
            return (None, None);
        }

        return self.case_2(seed_range);
    }

    fn case_2(&self, seed_range: &SeedRange) -> (Option<Vec<SeedRange>>, Option<SeedRange>)
    {     
        if seed_range.start > self.src_end
        {
            return (None, None);
        }
        return (None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_seed() {
        let test_map = SeedMap{src_start: 2, src_end: 7, dest_start: 10, dest_end: 14, size: 5};

        let test_seed = Seed{value: 1};
        assert_eq!(test_map.map_seed(&test_seed), None);

        let test_seed = Seed{value: 2};
        assert_eq!(test_map.map_seed(&test_seed), Some(Seed{value: 10}));

        let test_seed = Seed{value: 3};
        assert_eq!(test_map.map_seed(&test_seed), Some(Seed{value: 11}));

        let test_seed = Seed{value: 4};
        assert_eq!(test_map.map_seed(&test_seed), Some(Seed{value: 12}));

        let test_seed = Seed{value: 5};
        assert_eq!(test_map.map_seed(&test_seed), Some(Seed{value: 13}));

        let test_seed = Seed{value: 6};
        assert_eq!(test_map.map_seed(&test_seed), Some(Seed{value: 14}));

        let test_seed = Seed{value: 7};
        assert_eq!(test_map.map_seed(&test_seed), None);
    }

    #[test]
    fn map_seed_ptr() {
        let test_map = SeedMap{src_start: 2, src_end: 7, dest_start: 10, dest_end: 14, size: 5};

        let mut test_seed = Seed{value: 1};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, false);
        assert_eq!(test_seed.value, 1);

        let mut test_seed = Seed{value: 2};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, true);
        assert_eq!(test_seed.value, 10);

        let mut test_seed = Seed{value: 3};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, true);
        assert_eq!(test_seed.value, 11);

        let mut test_seed = Seed{value: 4};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, true);
        assert_eq!(test_seed.value, 12);

        let mut test_seed = Seed{value: 5};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, true);
        assert_eq!(test_seed.value, 13);

        let mut test_seed = Seed{value: 6};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, true);
        assert_eq!(test_seed.value, 14);

        let mut test_seed = Seed{value: 7};
        let result = unsafe { test_map.map_seed_ptr(&mut test_seed.value) };
        assert_eq!(result, false);
        assert_eq!(test_seed.value, 7);
    }
}