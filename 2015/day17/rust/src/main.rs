use clap::Parser;

use day17::part_a::part_a;
use day17::part_b::part_b;

#[derive(Parser)]
struct Cli {
    path: String,
    size: usize,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 17");
    let args:Cli = Cli::parse();

    println!("path: {:?}, size: {:?}, run: {:?}", args.path, args.size, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.path, &args.size),
        "part_b" => part_b(&args.path, &args.size),
        &_ => println!("Invalid run")
    }
}

