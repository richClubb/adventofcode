use clap::Parser;

use day24::part_a::part_a;

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
    bag_count: usize,
}

fn main() {
    println!("Advent of Code 2015 - Day 01");
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}, bag_count: {:?}", args.path, args.run, args.bag_count);

    match args.run.as_str() {
        "part_a" => part_a(&args.path, &args.bag_count),
        &_ => println!("Invalid run")
    }
}

