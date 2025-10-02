use::rayon::prelude::*;

use crate::seed::Seed;
use crate::seed_range::{SeedRange, get_seed_ranges_from_file};
use crate::seed_map_layer::{get_map_layers_from_file, map_seed_inverse, map_inverse_block_find_lowest_val};

pub fn part_b_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000000000);
    let map_layers = get_map_layers_from_file(&path);

    let min_value = seed_ranges.iter().map(|a| a.get_lowest_seed_in_range(&map_layers)).min().unwrap();

    println!("Part B forward brute force: {min_value}");
}

pub fn part_b_forward_ptr(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000000000);
    let map_layers = get_map_layers_from_file(&path);

    let min_value = seed_ranges.iter().map(|a| a.get_lowest_seed_in_range_ptr(&map_layers)).min().unwrap();

    println!("Part B forward brute force: {min_value}");
}


pub fn part_b_parallel_forward(path: &String){

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000);
    let map_layers = get_map_layers_from_file(&path);

    let result = seed_ranges.par_iter().map(|s| s.get_lowest_seed_in_range(&map_layers)).min().unwrap();

    println!("Part B forward parallel: {result}");

}

pub fn part_b_inverse(path: &String)
{

    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000000000);
    let map_layers = get_map_layers_from_file(&path);

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

pub fn part_b_parallel_inverse(path: &String)
{
    let seed_ranges:Vec<SeedRange> = get_seed_ranges_from_file(&path, 100000000);
    let map_layers = get_map_layers_from_file(&path);

    let blocksize:u64 = 1000000;
    let mut block:u64 = 0;
    let par_num:u64 = 20;

    loop {
        
        let results:Vec<u64> = (0..20)
                                                .into_par_iter()
                                                .map(|a:u64| map_inverse_block_find_lowest_val(block+(blocksize*a), block+((blocksize*a)+blocksize), &seed_ranges, &map_layers))
                                                .filter(|a| a.is_some())
                                                .map(|a| a.unwrap())
                                                .collect();

        // exit condition
        if results.len() > 0
        {
            println!("Part B inverse parallel: {}",results[0]);
            break;
        }

        block += blocksize * par_num;
    }

}

pub fn part_b_ranges(path: &String)
{
    let _:Vec<SeedRange> = get_seed_ranges_from_file(&path, 1000000000000);
    let _ = get_map_layers_from_file(&path);

    let _: Vec<SeedRange> = Vec::new();

}