use clap::Parser;

use day22::part_a::part_a;
use day22::part_b::part_b;

#[derive(Parser)]
struct Cli {
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 21");
    let args:Cli = Cli::parse();

    println!("run: {:?}", args.run);

    match args.run.as_str() {
        "part_a" => part_a(),
        "part_b" => part_b(),
        &_ => println!("Invalid run")
    }
}

