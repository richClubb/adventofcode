use crate::seed::Seed;
use crate::seed_map_layer::{SeedMapLayers};

use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;

pub struct SeedRange{
    pub start: u64,
    pub end: u64,
    pub size: u64
}

impl SeedRange {

    pub fn new(start: u64, size: u64) -> SeedRange {
        return SeedRange{start: start, end: start + size, size:size}
    }

    pub fn get_lowest_seed_in_range(&self, map_layers: &SeedMapLayers) -> u64
    {
        let mut min_value = std::u64::MAX;

        for seed_val in self.start..self.end
        {
            let result:Seed = map_layers.map_seed(&Seed{value: seed_val}).unwrap();
            if result.value < min_value
            {
                min_value = result.value;
            }
        }

        return min_value
    }

    pub fn get_lowest_seed_in_range_ptr(&self, map_layers: &SeedMapLayers) -> u64
    {
        let mut min_value = std::u64::MAX;

        for seed_val in self.start..self.end
        {
            let mut value = seed_val;
            let result = unsafe { map_layers.map_seed_ptr(&mut value) };
            if result < min_value
            {
                min_value = result;
            }
        }

        return min_value
    }
}

pub fn get_seed_ranges_from_file(path: &String, max_range_size: u64) -> Vec<SeedRange>
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

                let mut remaining = size;
                let mut curr_start = start;
                while remaining > 0 {
                    if remaining >= max_range_size {
                        seed_ranges.push(SeedRange::new(curr_start, max_range_size));
                        remaining -= max_range_size;
                        curr_start += max_range_size;
                    }
                    else {
                        seed_ranges.push(SeedRange::new(curr_start, remaining));
                        remaining = 0
                    }
                }
            }
        }
    }

    return seed_ranges
}