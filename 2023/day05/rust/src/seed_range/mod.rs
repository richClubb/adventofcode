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

pub fn get_lowest_seed_in_range(seed_range: &SeedRange, map_layers: &SeedMapLayers) -> u64
{
    let mut min_value = std::u64::MAX;

    for seed_val in seed_range.start..seed_range.end + 1
    {
        let result:Seed = map_layers.map_seed(&Seed{value: seed_val}).unwrap();
        if result.value < min_value
        {
            min_value = result.value;
        }
    }

    return min_value
}

pub fn get_lowest_seed_in_range_ptr(seed_range: &SeedRange, map_layers: &SeedMapLayers) -> u64
{
    let mut min_value = std::u64::MAX;

    for seed_val in seed_range.start..seed_range.end + 1
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