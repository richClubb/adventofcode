use clap::Parser;

use day10::part_a::part_a;
use day10::part_b::part_b;

#[derive(Parser)]
struct Cli {
    input: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 01");
    let args:Cli = Cli::parse();

    println!("input: {:?}, run: {:?}", args.input, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.input),
        "part_b" => part_b(&args.input),
        &_ => println!("Invalid run")
    }
}

