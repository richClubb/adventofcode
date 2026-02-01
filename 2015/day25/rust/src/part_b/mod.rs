use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut value = 0;

    for line in buf_reader.lines()
    {
        for (index, character) in line.unwrap().chars().enumerate() {
            match character {
                '(' => value = value + 1,
                ')' => value = value - 1,
                _ => println!("invalid character {}", character),
            }

            if value == -1
            {
                println!("basement entry at: {}", index + 1);
                return;
            }
        }
    }

    println!("Never entered the basement");
}