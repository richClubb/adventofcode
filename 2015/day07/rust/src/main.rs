use clap::Parser;

use day07::part_a::part_a;

#[derive(Parser)]
struct Cli {
    path: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 07");
    let args:Cli = Cli::parse();

    println!("path: {:?}", args.path);

    part_a(&args.path)
        
}