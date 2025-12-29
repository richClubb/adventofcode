use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::HashMap;

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut x: i64 = 0;
    let mut y: i64 = 0;

    let mut houses: HashMap<(i64, i64), u64> = HashMap::new();

    houses.insert((0, 0), 1);

    for line in buf_reader.lines()
    {
        match line {
            Ok(a) => {
                for character in a.chars() {
                    match character {
                        '^' => y += 1,
                        'v' => y -= 1,
                        '>' => x += 1,
                        '<' => x -= 1,
                        _ => x += 0,
                    };

                    let curr_pos = (x, y);

                    if !houses.contains_key(&curr_pos) {
                        houses.insert(curr_pos, 1);
                    }
                }
            },
            Err(_) => println!("Error in line"),
        };

        println!("Total: {}", houses.len())
    }
}

