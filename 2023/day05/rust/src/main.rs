use clap::Parser;
use rayon::prelude::*;

mod day5;

use crate::day5::day5::{
    Seed, 
    SeedRange, 
    MapLayer, 
    get_lowest_seed_in_range, 
    get_map_layers_from_file, 
    get_seed_ranges_from_file, 
    get_seeds_from_file,
    map_seed,
    map_seed_inverse,
    map_inverse_block_find_lowest_val
};

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

fn main() {
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    if &args.run == "part_a"
    {
        part_a(&args.path);
    }
    else if &args.run == "part_b_forward" {
        part_b_forward(&args.path);
    }
    else if &args.run == "part_b_inverse" {
        part_b_inverse(&args.path);
    }
    else if &args.run == "part_b_parallel_forward" {
        part_b_parallel_forward(&args.path);
    }
    else if &args.run == "part_b_parallel_inverse" {
        part_b_parallel_inverse(&args.path);
    }
    else if &args.run =="part_b_ranges" {
        part_b_ranges(&args.path)
    }
    else {
        println!("Unknown run");
    }
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

fn part_b_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let min_value = seed_ranges.iter().map(|a| get_lowest_seed_in_range(a, &map_layers)).min().unwrap();

    println!("Part B forward brute force: {min_value}");

}

fn part_b_parallel_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let result = seed_ranges.par_iter().map(|s| get_lowest_seed_in_range(s, &map_layers)).min().unwrap();

    println!("Part B forward parallel: {result}");

}

fn part_b_inverse(path: &String)
{

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let mut curr_val: u64 = 0;
    'main_iter: loop 
    {
        let result = map_seed_inverse(&Seed{value:curr_val}, &map_layers);

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

fn part_b_parallel_inverse(path: &String)
{
    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let blocksize:u64 = 1000;
    let mut block:u64 = 0;
    let par_num:u64 = 4;

    loop {
        
        //let results: Option<u64> = (0..par_num).into_par_iter().for_each(|a:u64| map_inverse_block_find_lowest_val(block*a, (block*a)+blocksize, &map_layers).unwrap());
        let results:Vec<u64> = (0..par_num)
                                                .into_par_iter()
                                                .map(|a:u64| map_inverse_block_find_lowest_val(block+(blocksize*a), block+((blocksize*a)+blocksize), &seed_ranges, &map_layers))
                                                .filter(|a| a.is_some())
                                                .map(|a| a.unwrap())
                                                .collect();
        
        // leaving this here just as reference
        // results.iter().for_each(|a| println!("{}", a));

        // exit condition
        if results.len() > 0
        {
            println!("Part B inverse parallel: {}",results[0]);
            break;
        }

        block += blocksize * par_num;
    }

}

fn part_b_ranges(path: &String)
{
    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path);
    let map_layers:Vec<MapLayer> = get_map_layers_from_file(&path);

    let mut result: Vec<SeedRange> = Vec::new();

}
