use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;
use rayon::prelude::*;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

struct Seed {
    value: u64
}

struct SeedRange{
    start: u64,
    end: u64,
    size: u64
}

struct Map {
    dest_start: u64,
    dest_end: u64,
    src_start: u64,
    src_end: u64,
    size: u64,
}

struct MapLayer {
    maps: Vec<Map>
}

fn main() {
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    if &args.run == "part_a"
    {
        part_a(&args.path);
    }
    else if &args.run == "part_b_forward" {
        part_b(&args.path);
    }
    else if &args.run == "part_b_inverse" {
        part_b_inverse(&args.path);
    }
    else if &args.run == "part_b_parallel" {
        part_b_parallel(&args.path);
    }
    else {
        println!("Unknown run");
    }
}

fn get_seeds_from_file(path: &String) -> Vec<Seed>
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

fn get_seed_ranges_from_file(path: &String) -> Vec<SeedRange>
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

fn get_map_layers_from_file(path: &String) -> Vec<MapLayer>
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

fn map_seed(seed: &Seed, map_layers: &Vec<MapLayer>) -> Seed
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

fn map_seed_inverse(seed: &Seed, map_layers: &Vec<MapLayer>) -> Seed
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

fn get_lowest_seed_in_range(seed_range: &SeedRange, map_layers: &Vec<MapLayer>) -> u64
{
    let mut min_value = std::u64::MAX;

    for seed_val in seed_range.start..seed_range.end + 1
    {
        let seed = Seed{value: seed_val};
        let result:Seed = map_seed(&seed, &map_layers);
        if result.value < min_value
        {
            min_value = result.value;
        }
    }

    return min_value
}


fn part_a(path: &String){

    let seeds:Vec<Seed> = get_seeds_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path); 

    let mut min_value = std::u64::MAX;
    for seed in seeds
    {
        let result:Seed = map_seed(&seed, &map_layers);
        if result.value < min_value
        {
            min_value = result.value;
        }
    }

    println!("Part A forward brute force: {min_value}");

}

fn part_b(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let mut min_value = std::u64::MAX;
    for seed_range in seed_ranges
    {
        for seed_val in seed_range.start..seed_range.end + 1
        {
            let seed = Seed{value: seed_val};
            let result:Seed = map_seed(&seed, &map_layers);
            if result.value < min_value
            {
                min_value = result.value;
            }
        }
    }

    println!("Part B forward brute force: {min_value}");

}

fn part_b_inverse(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let mut curr_val: u64 = 0;
    'main_iter: loop 
    {
        let seed = Seed{value:curr_val};

        let result = map_seed_inverse(&seed, &map_layers);

        for seed_range in &seed_ranges
        {
            if (result.value >= seed_range.start) && (result.value <= seed_range.end)
            {
                break 'main_iter;
            }
        }
        curr_val += 1;
    }

    println!("Part B inverse brute force: {curr_val}");

}

fn part_b_parallel(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let mut min_value = std::u64::MAX;
    let results: Vec<u64> = seed_ranges.par_iter().map(|s| get_lowest_seed_in_range(s, &map_layers)).collect();

    let mut min_value = std::u64::MAX;
    for result in results
    {
        if result < min_value
        {
            min_value = result;
        }
    }

    println!("Part B forward parallel: {min_value}");

}
