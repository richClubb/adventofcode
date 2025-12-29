
use crate::seed::{Seed, get_seeds_from_file};
use crate::seed_map_layer::{ get_map_layers_from_file};

pub fn part_a(path: &String){

    let seeds:Vec<Seed> = get_seeds_from_file(&path);
    let map_layers = get_map_layers_from_file(&path); 

    let mut min_value = std::u64::MAX;
    for seed in seeds
    {
        let result:Seed = map_layers.map_seed(&seed).unwrap();
        
        if result.value < min_value
        {
            min_value = result.value;
        }
    }

    println!("Part A forward brute force: {min_value}");
}

pub fn part_a_ptr(path: &String){

    let seeds:Vec<Seed> = get_seeds_from_file(&path);
    let map_layers = get_map_layers_from_file(&path); 

    let mut min_value = std::u64::MAX;
    for seed in seeds
    {
        let mut seed_value = seed.value;
        let result = unsafe { map_layers.map_seed_ptr(&mut seed_value) };
        
        if result < min_value
        {
            min_value = result;
        }
    }

    println!("Part A forward brute force ptr version: {min_value}");
}