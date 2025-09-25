use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;

#[derive(Debug)]
#[derive(PartialEq)]
pub struct Seed {
    pub value: u64
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