use clap::Parser;

use day04::part_a::part_a;
use day04::part_b::part_b;

#[derive(Parser)]
struct Cli {
    key: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 04");
    let args:Cli = Cli::parse();

    println!("key: {:?}, run {:?}", args.key, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.key),
        "part_b" => part_b(&args.key),
        &_ => println!("Invalid run")
    }
}