mod part_a;
mod part_b;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];

    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");

    let part_a_result = part_a::part_a(contents.clone());
    let part_b_result = part_b::part_b(contents.clone());

    println!("Day 1 Part A: {part_a_result}");
    println!("Day 1 Part A: {part_b_result}");
}
