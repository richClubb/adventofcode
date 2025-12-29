use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::HashMap;

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut santa_x: i64 = 0;
    let mut santa_y: i64 = 0;

    let mut robot_x: i64 = 0;
    let mut robot_y: i64 = 0;

    let mut houses: HashMap<(i64, i64), u64> = HashMap::new();

    for line in buf_reader.lines()
    {
        match line {
            Ok(a) => {
                for (index, character) in a.chars().enumerate() {
                    
                    if index == 0 { houses.insert((0, 0), 1); };

                    if ((index % 2) != 0)
                    {
                        match character {
                            '^' => santa_y += 1,
                            'v' => santa_y -= 1,
                            '>' => santa_x += 1,
                            '<' => santa_x -= 1,
                            _ => santa_x += 0,
                        };

                        let curr_pos = (santa_x, santa_y);

                        if !houses.contains_key(&curr_pos) {
                            houses.insert(curr_pos, 1);
                        }
                    }
                    else
                    {
                        match character {
                            '^' => robot_y += 1,
                            'v' => robot_y -= 1,
                            '>' => robot_x += 1,
                            '<' => robot_x -= 1,
                            _ => robot_x += 0,
                        };

                        let curr_pos = (robot_x, robot_y);

                        if !houses.contains_key(&curr_pos) {
                            houses.insert(curr_pos, 1);
                        }
                    }
                }
            },
            Err(_) => println!("Error in line"),
        };

        println!("Total: {}", houses.len())
    }
}

