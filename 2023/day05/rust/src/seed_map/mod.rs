use crate::seed_range::SeedRange;

pub struct SeedMap {
    pub dest_start: u64,
    pub dest_end: u64,
    pub src_start: u64,
    pub src_end: u64,
    pub size: u64,
}

impl SeedMap {

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