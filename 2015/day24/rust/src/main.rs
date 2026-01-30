use clap::Parser;

use day24::part_a::part_a;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 01");
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.path),
        &_ => println!("Invalid run")
    }
}

