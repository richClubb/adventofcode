use crate::instruction::Instruction;
use crate::location::Location;

use std::cmp::max;

use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;

fn decode_line(line: &String) -> (Instruction, Location, Location) {

    let re = Regex::new(r"(?<inst>(turn\son)|(toggle)|(turn\soff))\s(?<start>[0-9]{1,3}\,[0-9]{1,3})\sthrough\s(?<end>[0-9]{1,3}\,[0-9]{1,3})").unwrap();

    re.captures(line).map(|caps| {
            let inst = Instruction::new(String::from(caps.name("inst").unwrap().as_str()));
            let start = Location::new(String::from(caps.name("start").unwrap().as_str()));
            let end = Location::new(String::from(caps.name("end").unwrap().as_str()));
            (inst, start, end)
        }
    ).unwrap()
}

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut lights = vec![vec![0; 1000]; 1000];

    for line in buf_reader.lines() {
        let (curr_inst, start, end) = decode_line(&line.unwrap());

        for x_index in start.x_pos..=end.x_pos {
            for y_index in start.y_pos..=end.y_pos {
                match curr_inst {
                    Instruction::TurnOn => lights[x_index][y_index] += 1,
                    Instruction::TurnOff => lights[x_index][y_index] = max(lights[x_index][y_index] - 1, 0),
                    Instruction::Toggle => lights[x_index][y_index] += 2,
                    Instruction::Invalid => (),
                }
            }
        }
    }

    let mut intensity = 0;

    for x_index in 0..1000 {
        for y_index in 0..1000 {
            intensity +=  lights[x_index][y_index];
        }
    }

    println!("{}", intensity);

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_line() {
        assert_eq!(decode_line(&String::from("turn on 0,0 through 999,999")), (Instruction::TurnOn, Location{x_pos: 0, y_pos: 0}, Location{x_pos: 999, y_pos: 999}));
        assert_eq!(decode_line(&String::from("toggle 0,0 through 999,0")), (Instruction::Toggle, Location{x_pos: 0, y_pos: 0}, Location{x_pos: 999, y_pos: 0}));
        assert_eq!(decode_line(&String::from("turn off 499,499 through 500,500")), (Instruction::TurnOff, Location{x_pos: 499, y_pos: 499}, Location{x_pos: 500, y_pos: 500}));
    }
}