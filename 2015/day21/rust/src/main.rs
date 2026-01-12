use clap::Parser;

use day21::part_a::part_a;
use day21::part_b::part_b;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 21");
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.path),
        "part_b" => part_b(&args.path),
        &_ => println!("Invalid run")
    }
}

