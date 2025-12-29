use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut pos: i16 = 50;
    let mut password = 0;

    for line in buf_reader.lines()
    {
        let increment = line.as_ref().unwrap()[1..].parse::<i16>().unwrap();
        let direction = line.as_ref().unwrap().chars().nth(0).unwrap();

        match direction {
            'L' => pos = (pos - increment) % 100,
            'R' => pos = (pos + increment) % 100,
            _ => println!("error"),
        };

        if pos == 0 { password += 1 }
    }

    println!("{}", password);
}