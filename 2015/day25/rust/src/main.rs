use clap::Parser;

use day25::part_a::part_a;
use day25::part_b::part_b;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 25");
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.path, 2978, 3083),
        "part_b" => part_b(&args.path),
        &_ => println!("Invalid run")
    }
}

