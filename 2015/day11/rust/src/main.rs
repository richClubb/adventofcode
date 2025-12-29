use clap::Parser;

use day11::part_a::part_a;

#[derive(Parser)]
struct Cli {
    input: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 11");
    let args:Cli = Cli::parse();

    println!("input: {:?}", args.input);

    part_a(&args.input);
    
}

