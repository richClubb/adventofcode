
use crate::seed::{Seed, get_seeds_from_file};
use crate::seed_map_layer::{SeedMapLayer, get_map_layers_from_file, map_seed};

pub fn part_a(path: &String){

    let seeds:Vec<Seed> = get_seeds_from_file(&path);
    let map_layers:Vec<SeedMapLayer> = get_map_layers_from_file(&path); 

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