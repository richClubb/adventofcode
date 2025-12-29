use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut pos: i16 = 50;
    let mut password = 0;

    for line in buf_reader.lines()
    {
        let raw_increment = line.as_ref().unwrap()[1..].parse::<i16>().unwrap();
        let direction = line.as_ref().unwrap().chars().nth(0).unwrap();

        if raw_increment > 100 {
            let num_turns = raw_increment / 100;
            password += num_turns;
        }

        let increment = raw_increment % 100;

        let increment_vector = match direction {
            'L' => - increment,
            'R' => increment,
            _ => 0,
        };

        if ( pos == 0 ) || ( pos == 100 ) {
            password += 1;

            if increment_vector < 0 {
                pos = 100 + increment_vector;
            }
            else
            {
                pos = 0 + increment_vector;
            }

            continue;
        }
        
        let new_pos = pos + increment_vector;

        if new_pos < 0 {
            password += 1;
            pos = 100 + new_pos;
        }
        else if new_pos > 100 {
            password += 1;
            pos = new_pos - 100;
        }
        else {
            pos = new_pos;
        }
    }

    println!("{}", password);
}