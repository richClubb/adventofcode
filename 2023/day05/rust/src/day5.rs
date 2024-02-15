
pub mod day5{

    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use regex::Regex;

    pub struct Seed {
        pub value: u64
    }

    pub struct SeedRange{
        pub start: u64,
        pub end: u64,
        pub size: u64
    }

    pub struct Map {
        pub dest_start: u64,
        pub dest_end: u64,
        pub src_start: u64,
        pub src_end: u64,
        pub size: u64,
    }

    impl Map {

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

    pub struct MapLayer {
        maps: Vec<Map>
    }

    impl MapLayer {

        fn map_seed_ranges(seed_ranges: &Vec<SeedRange>) -> Vec<SeedRange>
        {
            let mut result: Vec<SeedRange> = Vec::new();
            
            

            return result;
        }
    }

    pub fn get_seeds_from_file(path: &String) -> Vec<Seed>
    {
        let mut seeds: Vec<Seed> = Vec::new();

        let file: File = File::open(path).expect("Could not open file");
        let buf_reader:BufReader<File> = BufReader::new(file);

        let seed_regex: Regex = Regex::new(r"seeds\:\s([0-9\s]{1,})").unwrap();

        // parse the file and get the seeds
        for line in buf_reader.lines()
        {
            if let Some(seeds_raw) = seed_regex.captures(&line.unwrap())
            {
                let seed_values: Vec<u64> = seeds_raw[1].split_whitespace().map(|s| s.parse().unwrap()).collect();
                for seed_value in seed_values
                {
                    seeds.push(Seed{value:seed_value});
                }
            }
        }

        return seeds
    }

    pub fn get_seed_ranges_from_file(path: &String) -> Vec<SeedRange>
    {
        let mut seed_ranges: Vec<SeedRange> = Vec::new();
        let file: File = File::open(path).expect("Could not open file");
        let buf_reader = BufReader::new(file);

        let seed_regex: Regex = Regex::new(r"seeds\:\s([0-9\s]{1,})").unwrap();

        // parse the file and get the seeds
        for line in buf_reader.lines()
        {
            if let Some(seeds_raw) = seed_regex.captures(&line.unwrap())
            {
                let seed_values: Vec<u64> = seeds_raw[1].split_whitespace().map(|s| s.parse().unwrap()).collect();
                for seed_index in 0..seed_values.len() / 2
                {
                    let start = seed_values[seed_index*2];
                    let size = seed_values[seed_index*2+1];
                    let end = start+size-1;
                    
                    seed_ranges.push(SeedRange {start: start, end: end, size:size});
                }
            }
        }

        return seed_ranges
    }

    pub fn get_map_layers_from_file(path: &String) -> Vec<MapLayer>
    {
        let mut map_layers: Vec<MapLayer> = Vec::new();

        let file: File = File::open(path).expect("Could not open file");
        let buf_reader = BufReader::new(file);

        
        let seed_regex: Regex = Regex::new(r"seeds\:\s([0-9\s]{1,})").unwrap();
        let map_title_regex: Regex = Regex::new(r"[a-z]{1,}\-to\-[a-z]{1,}\smap\:").unwrap();
        let map_regex: Regex = Regex::new(r"([0-9]{1,})\s([0-9]{1,})\s([0-9]{1,})").unwrap();

        let mut curr_maps: Vec<Map> = Vec::new();
        let mut started: bool = false;
        // parse the file and get the seeds
        for line in buf_reader.lines()
        {
            if seed_regex.is_match(&line.as_ref().unwrap())
            {
                continue;
            } 

            if let Some(map_raw) = map_regex.captures(&line.as_ref().unwrap())
            {
                let dest: u64 = map_raw[1].parse().unwrap();
                let src: u64 = map_raw[2].parse().unwrap();
                let size: u64 = map_raw[3].parse().unwrap();
                curr_maps.push(Map{dest_start: dest, dest_end: dest+ size -1, src_start: src, src_end: src+size-1, size:size});
            }

            if map_title_regex.is_match(&line.as_ref().unwrap())
            {
                if started == false
                {
                    started = true;
                }
                else {
                    map_layers.push(MapLayer{maps:curr_maps});
                    curr_maps = Vec::new();
                }
                continue;
            }
        }

        map_layers.push(MapLayer{maps:curr_maps});

        return map_layers
    } 

    pub fn map_seed(seed: &Seed, map_layers: &Vec<MapLayer>) -> Seed
    {

        let mut result: Seed = Seed{ value: seed.value};
        for map_layer in map_layers
        {
            for map in &map_layer.maps
            {
                if (result.value >= map.src_start) && (result.value <= map.src_end)
                {
                    result.value = map.dest_start + result.value - map.src_start;
                    break;
                }
            }
        }

        return result;
    }

    pub fn map_seed_inverse(seed: &Seed, map_layers: &Vec<MapLayer>) -> Seed
    {

        let mut result: Seed = Seed{ value: seed.value};
        for map_layer in map_layers.iter().rev()
        {
            for map in &map_layer.maps
            {
                if (result.value >= map.dest_start) && (result.value <= map.dest_end)
                {
                    result.value = map.src_start + result.value - map.dest_start;
                    break;
                }
            }
        }

        return result;
    }

    pub fn map_inverse_block_find_lowest_val(start: u64, end: u64, seed_ranges: &Vec<SeedRange>, map_layers: &Vec<MapLayer>) -> Option<u64>
    {
        
        for value in start..end
        {

            let result = map_seed_inverse(&Seed{value:value}, &map_layers);

            for seed_range in seed_ranges
            {
                if (result.value >= seed_range.start) && (result.value <= seed_range.end)
                {
                    return Some(value);
                }
            }
        }

        return None;
    }


    pub fn get_lowest_seed_in_range(seed_range: &SeedRange, map_layers: &Vec<MapLayer>) -> u64
    {
        let mut min_value = std::u64::MAX;

        for seed_val in seed_range.start..seed_range.end + 1
        {
            let result:Seed = map_seed(&Seed{value: seed_val}, &map_layers);
            if result.value < min_value
            {
                min_value = result.value;
            }
        }

        return min_value
    }
}

#[cfg(test)]
mod tests
{
    use super::day5;
    
    #[test]
    fn test_case_1()
    {
        let seed_range = day5::SeedRange{ start: 1, end: 3, size: 3};
        let map = day5::Map{src_start: 5, src_end: 7, dest_start: 10, dest_end: 12, size: 2};

        let (mapped, unmapped) = map.map_seed_range(&seed_range);

        assert_eq!(mapped.unwrap().len(), 1);
    }

}