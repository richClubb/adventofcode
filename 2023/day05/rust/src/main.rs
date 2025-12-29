use clap::Parser;

use day5::part_a::{part_a, part_a_ptr};
use day5::part_b::{part_b_forward, part_b_forward_ptr, part_b_inverse, part_b_parallel_forward, part_b_parallel_inverse, part_b_ranges};

#[derive(Parser)]
struct Cli {
    path: String,
    run: String,
}

fn main() {
    let args:Cli = Cli::parse();

    println!("path: {:?}, run: {:?}", args.path, args.run);

    if &args.run == "part_a"
    {
        part_a(&args.path);
    }
    else if &args.run == "part_a_ptr"
    {
        part_a_ptr(&args.path);
    }
    else if &args.run == "part_b_forward" {
        part_b_forward(&args.path);
    }
    else if &args.run == "part_b_forward_ptr" {
        part_b_forward_ptr(&args.path);
    }
    else if &args.run == "part_b_inverse" {
        part_b_inverse(&args.path);
    }
    else if &args.run == "part_b_parallel_forward" {
        part_b_parallel_forward(&args.path);
    }
    else if &args.run == "part_b_parallel_inverse" {
        part_b_parallel_inverse(&args.path);
    }
    else if &args.run =="part_b_ranges" {
        part_b_ranges(&args.path)
    }
    else {
        println!("Unknown run");
    }
}

