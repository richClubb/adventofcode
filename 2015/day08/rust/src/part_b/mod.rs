use std::fs::File;
use std::io::{BufRead, BufReader};

fn calculate_line_chars(input: &String) -> usize {
    
    let mut new_string = String::new();
    
    new_string.push('"');

    for char in input.chars() {
        match char {
            '"' => new_string.push_str("\\\""),
            '\\' => new_string.push_str("\\\\"),
            _ => new_string.push(char),
        }
    }

    new_string.push('"');

    return new_string.len();
}


pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut char_total = 0;
    let mut encoded_total = 0;

    for line in buf_reader.lines() {
        let line = line.unwrap();
        char_total += &line.len();

        encoded_total += calculate_line_chars(&line);

    }

    println!("Total chars: {char_total}");
    println!("Total chars: {encoded_total}");

    println!("{}", encoded_total - char_total);
}