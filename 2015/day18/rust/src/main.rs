use clap::Parser;

use day18::part_a::part_a;
use day18::part_b::part_b;

#[derive(Parser)]
struct Cli {
    path: String,
    steps: usize,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 18");
    let args:Cli = Cli::parse();

    println!("path: {:?}, steps: {:?}, run: {:?}", args.path, args.steps, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.path, &args.steps),
        "part_b" => part_b(&args.path, &args.steps),
        &_ => println!("Invalid run")
    }
}

