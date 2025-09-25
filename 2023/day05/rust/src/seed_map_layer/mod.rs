use crate::seed::Seed;
use crate::seed_map::SeedMap;
use crate::seed_range::SeedRange;

use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;

pub struct SeedMapLayer {
    maps: Vec<SeedMap>
}

impl SeedMapLayer {

    // fn map_seed_ranges(seed_ranges: &Vec<SeedRange>) -> Vec<SeedRange>
    // {
    //     let _: Vec<SeedRange> = Vec::new();
        
        

    //     return result;
    // }
}

pub fn get_map_layers_from_file(path: &String) -> Vec<SeedMapLayer>
{
    let mut map_layers: Vec<SeedMapLayer> = Vec::new();

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader = BufReader::new(file);

    
    let seed_regex: Regex = Regex::new(r"seeds\:\s([0-9\s]{1,})").unwrap();
    let map_title_regex: Regex = Regex::new(r"[a-z]{1,}\-to\-[a-z]{1,}\smap\:").unwrap();
    let map_regex: Regex = Regex::new(r"([0-9]{1,})\s([0-9]{1,})\s([0-9]{1,})").unwrap();

    let mut curr_maps: Vec<SeedMap> = Vec::new();
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
            curr_maps.push(SeedMap{dest_start: dest, dest_end: dest+ size -1, src_start: src, src_end: src+size-1, size:size});
        }

        if map_title_regex.is_match(&line.as_ref().unwrap())
        {
            if started == false
            {
                started = true;
            }
            else {
                map_layers.push(SeedMapLayer{maps:curr_maps});
                curr_maps = Vec::new();
            }
            continue;
        }
    }

    map_layers.push(SeedMapLayer{maps:curr_maps});

    return map_layers
}

pub fn map_seed(seed: &Seed, map_layers: &Vec<SeedMapLayer>) -> Seed
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

pub fn map_seed_inverse(seed: &Seed, map_layers: &Vec<SeedMapLayer>) -> Seed
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

pub fn map_inverse_block_find_lowest_val(start: u64, end: u64, seed_ranges: &Vec<SeedRange>, map_layers: &Vec<SeedMapLayer>) -> Option<u64>
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