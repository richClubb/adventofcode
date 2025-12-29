use crate::instruction::Instruction;
use crate::location::Location;

use std::fs::File;
use std::io::{BufRead, BufReader};
use regex::Regex;

fn decode_line(line: &String) -> (Instruction, Location, Location) {

    let re = Regex::new(r"(?<inst>(turn\son)|(toggle)|(turn\soff))\s(?<start>[0-9]{1,3}\,[0-9]{1,3})\sthrough\s(?<end>[0-9]{1,3}\,[0-9]{1,3})").unwrap();

    let results: (&str, &str, &str) = re.captures(line).map(|caps| {
            let inst = caps.name("inst").unwrap().as_str();
            let start = caps.name("start").unwrap().as_str();
            let end = caps.name("end").unwrap().as_str();
            (inst, start, end)
        }
    ).unwrap(); 

    (
        Instruction::new(String::from(results.0)), 
        Location::new(String::from(results.1)), 
        Location::new(String::from(results.2))
    )
}

fn toggle(input: usize) -> usize
{
    if input == 0 {
        return 1;
    }

    return 0;
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut lights = vec![vec![0; 1000]; 1000];

    for line in buf_reader.lines() {
        let action: (Instruction, Location, Location) = decode_line(&line.unwrap());

        let curr_inst = action.0;
        let start = action.1;
        let end = action.2;

        for x_index in start.x_pos..=end.x_pos {
            for y_index in start.y_pos..=end.y_pos {
                match curr_inst {
                    Instruction::TurnOn => lights[x_index][y_index] = 1,
                    Instruction::TurnOff => lights[x_index][y_index] = 0,
                    Instruction::Toggle => lights[x_index][y_index] = toggle(lights[x_index][y_index]),
                    Instruction::Invalid => (),
                }
            }
        }
    }

    let mut count = 0;

    for x_index in 0..1000 {
        for y_index in 0..1000 {
            if lights[x_index][y_index] == 1 {
                count += 1;
            }
        }
    }

    println!("{}", count);

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

    #[test]
    fn test_toggle() {
        assert_eq!(toggle(0), 1);
        assert_eq!(toggle(1), 0);
    }
}