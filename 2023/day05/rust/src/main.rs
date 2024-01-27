use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use dict::Dict;
use regex::Regex;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

struct Map {
    dest: u64,
    start: u64,
    size: u64,
}

fn main() {
    let args = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    part_a(args.path);

    //part_b(args.path);

}

fn process_file(path: String)
{
    let file = File::open(path).expect("Could not open file");

    let buf_reader = BufReader::new(file);

    let seed_regex = Regex::new(r"seeds\:\s([0-9\s]{1,})").unwrap();
    let map_title_regex = Regex::new(r"[a-z\-]{1,}\smap\:").unwrap();
    let map_regex = Regex::new(r"[0-9]{1,}").unwrap();

    let mut seeds = Vec::<u64>::new();
    let mut map_list = Vec::<Map>::new();
    let mut maps_dict = Dict::<u16>::new();

    for line in buf_reader.lines()
    {
        println!("Found line {}", line.expect("Could not read line"));
        


    }

}

fn part_a(path: String){

    process_file(path)
    
}

fn _part_b(_path: String){



}
