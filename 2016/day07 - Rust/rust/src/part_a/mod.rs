use std::fs::File;
use std::io::{BufRead, BufReader};

fn contains_abba(input: &String) -> bool {

    for index in 1..input.len() - 2{
        let curr_char = input.chars().nth(index).unwrap();
        let next_char = input.chars().nth(index + 1).unwrap();

        if curr_char == next_char {
            let prefix_char = input.chars().nth(index - 1).unwrap();
            let suffix_char = input.chars().nth(index + 2).unwrap();

            if prefix_char == curr_char {
                return false;
            }

            if prefix_char == suffix_char {
                return true;
            }
        }
    }

    return false
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let result: Vec<bool> = buf_reader.lines().map(|line| {
            let line = line.unwrap();

            let mut match_in_hypernet= false;
            let mut success = false;
            let mut buffer = String::new();

            for char in line.chars() {
                if char == '[' {
                    if success == false {
                        success = contains_abba(&buffer);
                    }
                    buffer.clear();
                    continue;
                }

                if char == ']' {
                    match_in_hypernet = contains_abba(&buffer);
                    if match_in_hypernet == true {
                        break;
                    }
                    buffer.clear();
                    continue;
                }

                buffer.push(char);
            }

            success = if match_in_hypernet {
                false
            } else {
                let result = if success {
                    true
                } else {
                    contains_abba(&buffer)
                };
                result
            };

            success
        }
    ).filter(|entry| entry == &true).collect();

    println!("{}", result.len());
}