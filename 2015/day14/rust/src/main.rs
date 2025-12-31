use clap::Parser;

use day14::part_a::part_a;
use day14::part_b::part_b;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
    length: usize,
}

fn main() {
    println!("Advent of Code 2015 - Day 14");
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}, length: {:?}", args.path, args.run, args.length);

    match args.run.as_str() {
        "part_a" => part_a(&args.path, args.length),
        "part_b" => part_b(&args.path, args.length),
        &_ => println!("Invalid run")
    }
}

