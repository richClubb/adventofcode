use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::HashMap;

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let houses = buf_reader.lines().fold(
        ((0, 0), HashMap::from([((0, 0), 1)])), 
        |mut acc, line| {
            match line {
                Ok(a_line) => {
                    for character in a_line.chars() {

                        let curr_pos = acc.0;
                        let curr_pos = match character {
                            '^' => (curr_pos.0, curr_pos.1 + 1),
                            'v' => (curr_pos.0, curr_pos.1 - 1),
                            '>' => (curr_pos.0 + 1, curr_pos.1),
                            '<' => (curr_pos.0 - 1, curr_pos.1),
                            _ => (curr_pos.0, curr_pos.1),
                        };

                        if !acc.1.contains_key(&curr_pos) {
                            acc.1.insert(curr_pos, 1);
                        }
                        acc.0 = curr_pos
                    }
                    acc
                },
                Err(_) => acc,
            }
        }
    );

    println!("Total: {}", houses.1.len())
}

