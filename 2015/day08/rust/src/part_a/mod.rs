use std::fs::File;
use std::io::{BufRead, BufReader};

fn calculate_line_chars(input: &String) -> usize {

    let mut count = 0;

    let stripped_line = &   input[1..input.len() - 1];

    let mut skip_char = 0;

    for (index, char) in stripped_line.chars().enumerate() {

        if skip_char > 0 {
            skip_char -= 1;
            continue;
        }

        if char == '\\' {
            let next_char = stripped_line.chars().nth(index+1).unwrap();
            match next_char {
                '\\' => {
                    skip_char = 1;
                    count += 1;
                    continue;
                },
                '"' => {
                    skip_char = 1;
                    count += 1;
                    continue;
                }
                'x' => {
                    skip_char = 3;
                    count += 1;
                    continue;
                }
                _ => ()
            }
        }

        count += 1;
    }

    return count;
}


pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut char_total = 0;
    let mut stripped_total = 0;

    for line in buf_reader.lines() {
        let line = line.unwrap();
        char_total += &line.len();

        stripped_total += calculate_line_chars(&line);

    }

    println!("Total chars: {char_total}");
    println!("Total chars: {stripped_total}");

    println!("{}", char_total - stripped_total);
}