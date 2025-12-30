use clap::Parser;

use day13::part_a::part_a;

#[derive(Parser)]
struct Cli {
    path: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 13");
    let args:Cli = Cli::parse();

    println!("path: {:?}", args.path);

    part_a(&args.path);

}

