use clap::Parser;

use day16::part_a::part_a;
use day16::part_b::part_b;

#[derive(Parser)]
struct Cli {
    sue_list: String,
    sue_info: String,
    run: String,
}

fn main() {
    println!("Advent of Code 2015 - Day 16");
    let args:Cli = Cli::parse();

    println!("sue list: {:?}, sue info: {:?}, run: {:?}", args.sue_list, args.sue_info, args.run);

    match args.run.as_str() {
        "part_a" => part_a(&args.sue_list, &args.sue_info),
        "part_b" => part_b(&args.sue_list, &args.sue_info),
        &_ => println!("Invalid run")
    }
}

