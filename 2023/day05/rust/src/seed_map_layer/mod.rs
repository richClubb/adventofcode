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

    pub fn map_seed(&self, seed: &Seed) -> Option<Seed> {

        for map in &self.maps {
            let result = map.map_seed(&seed);
            if result.is_some() 
            {
                return result;
            }
        };

        return None;
    }

    pub fn map_seed_ptr(&self, seed_value: *mut u64) -> bool {

        for map in &self.maps {
            let result = unsafe { map.map_seed_ptr(seed_value) };
            if result 
            {
                return true;
            }
        };

        return false;
    }
}

pub struct SeedMapLayers {
    pub layers: Vec<SeedMapLayer>
}

impl SeedMapLayers {

    pub fn map_seed(&self, seed: &Seed) -> Option<Seed>{
        let mut result = Seed{value: seed.value};
        for layer in &self.layers {
            let layer_result = layer.map_seed(&result);
            if layer_result.is_some() {
                result = layer_result.unwrap();
            }   
        }

        return Some(result);
    }

    pub unsafe fn map_seed_ptr(&self, seed_value: *mut u64) -> u64{
        for layer in &self.layers {
            let _ = layer.map_seed_ptr(seed_value);
        }

        return *seed_value;
    }
}

pub fn get_map_layers_from_file(path: &String) -> SeedMapLayers
{
    let mut layers: Vec<SeedMapLayer> = Vec::new();

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
            curr_maps.push(SeedMap::new(src, dest, size));
        }

        if map_title_regex.is_match(&line.as_ref().unwrap())
        {
            if started == false
            {
                started = true;
            }
            else {
                layers.push(SeedMapLayer{maps:curr_maps});
                curr_maps = Vec::new();
            }
            continue;
        }
    }

    layers.push(SeedMapLayer{maps:curr_maps});

    return SeedMapLayers{ layers: layers };
}

pub fn map_seed(seed: &Seed, map_layers: &Vec<SeedMapLayer>) -> Seed
{
    let mut result: Seed = Seed{ value: seed.value};
    for map_layer in map_layers
    {
        let inner_result = map_layer.map_seed(&result);
        if inner_result.is_some()
        {
            result = inner_result.unwrap();
            continue;
        }
    }

    return result;
}

pub fn map_seed_inverse(seed: &Seed, map_layers: &SeedMapLayers) -> Seed
{

    let mut result: Seed = Seed{ value: seed.value};
    for map_layer in map_layers.layers.iter().rev()
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

pub fn map_inverse_block_find_lowest_val(start: u64, end: u64, seed_ranges: &Vec<SeedRange>, map_layers: &SeedMapLayers) -> Option<u64>
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_seed_for_layer() {

        let test_map_1 = SeedMap{src_start: 2, src_end: 7, dest_start: 10, dest_end: 14, size: 5};
        let test_map_2 = SeedMap{src_start: 20, src_end: 27, dest_start: 50, dest_end: 57, size: 7};

        let test_map_layer = SeedMapLayer{maps: vec![test_map_1, test_map_2]};

        let test_seed = Seed{value: 1};
        assert_eq!(test_map_layer.map_seed(&test_seed), None);

        let test_seed = Seed{value: 2};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 10}));

        let test_seed = Seed{value: 3};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 11}));

        let test_seed = Seed{value: 4};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 12}));

        let test_seed = Seed{value: 5};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 13}));

        let test_seed = Seed{value: 6};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 14}));

        let test_seed = Seed{value: 7};
        assert_eq!(test_map_layer.map_seed(&test_seed), None);

        let test_seed = Seed{value: 19};
        assert_eq!(test_map_layer.map_seed(&test_seed), None);

        let test_seed = Seed{value: 20};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 50}));

        let test_seed = Seed{value: 21};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 51}));

        let test_seed = Seed{value: 22};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 52}));

        let test_seed = Seed{value: 23};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 53}));

        let test_seed = Seed{value: 24};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 54}));

        let test_seed = Seed{value: 25};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 55}));

        let test_seed = Seed{value: 26};
        assert_eq!(test_map_layer.map_seed(&test_seed), Some(Seed{value: 56}));

        let test_seed = Seed{value: 27};
        assert_eq!(test_map_layer.map_seed(&test_seed), None);

    }

    #[test]
    fn map_seed_for_layers() {

        let test_map_1 = SeedMap{src_start: 2, src_end: 7, dest_start: 10, dest_end: 14, size: 5};
        let test_map_2 = SeedMap{src_start: 20, src_end: 27, dest_start: 50, dest_end: 57, size: 7};

        let test_map_layer_1 = SeedMapLayer{maps: vec![test_map_1, test_map_2]};

        let test_map_3 = SeedMap{src_start: 52, src_end: 54, dest_start: 30, dest_end: 32, size: 2};

        let test_map_layer_2 = SeedMapLayer{maps: vec![test_map_3]};

        let test_map_layers = SeedMapLayers{layers: vec![test_map_layer_1, test_map_layer_2]};

        let value = Seed{value: 1};
        assert_eq!(test_map_layers.map_seed(&value), Some(Seed{value:1}));

        let value = Seed{value: 2};
        assert_eq!(test_map_layers.map_seed(&value), Some(Seed{value:10}));

        let value = Seed{value: 20};
        assert_eq!(test_map_layers.map_seed(&value), Some(Seed{value:50}));

        let value = Seed{value: 22};
        assert_eq!(test_map_layers.map_seed(&value), Some(Seed{value:30}));


    }
}