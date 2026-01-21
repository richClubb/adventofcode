use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

#[derive(Debug)]
struct Instruction {
    instruction: String,
    options: String,
}

impl Instruction {
    pub fn new(input: &String) -> Self {
        let input_re = Regex::new(r"(?<instruction>[a-z]{3})\s(?<options>[a-z,\+\-0-9\s]{1,})").unwrap();

        let (instruction, options) = input_re.captures(input).map(|caps| {
            let instruction = caps.name("instruction").unwrap().as_str().to_string();
            let options = caps.name("options").unwrap().as_str().to_string();
            (instruction, options)
        }).unwrap();

        return Instruction { instruction: instruction, options: options };
    }

    pub fn interpret(&self, a: usize, b: usize) -> (usize, usize, Option<isize>) {

        // println!("    {a}, {b}");
        let result = match self.instruction.as_str() {
            "hlf" => {
                // println!("  HLF");
                match self.options.as_str() {
                    "a" => (a / 2, b, None),
                    "b" => (a, b / 2, None),
                    _ => (a, b, None),
                }
            },
            "tpl" => {
                // println!("  TPL");
                match self.options.as_str() {
                    "a" => (a * 3, b, None),
                    "b" => (a, b * 3, None),
                    _ => (a, b, None),
                }
            },
            "inc" => {
                // println!("  INC");
                match self.options.as_str() {
                    "a" => (a + 1, b, None),
                    "b" => (a, b + 1, None),
                    _ => (a, b, None),
                }
            },
            "jmp" => {
                // println!("  JMP");
                let offset = self.options.parse::<isize>().unwrap();
                (a, b, Some(offset))
            },
            "jie" => {
                // println!("  JIE");
                let vals: Vec<&str> = self.options.split(", ").collect();
                match vals[0] {
                    "a" => {
                        let result = if a % 2 == 0 {
                            (a, b, Some(vals[1].parse::<isize>().unwrap()))
                        }
                        else {
                            (a, b, None)
                        };
                        result
                    }
                    "b" => {
                        let result = if a % 2 == 0 {
                            (a, b, Some(vals[1].parse::<isize>().unwrap()))
                        }
                        else {
                            (a, b, None)
                        };
                        result
                    }
                    _ => (0, 0, None)
                }
            },
            "jio" => {
                // println!("  JIO");
                let vals: Vec<&str> = self.options.split(", ").collect();
                match vals[0] {
                    "a" => {
                        let result = if a == 1 {
                            (a, b, Some(vals[1].parse::<isize>().unwrap()))
                        }
                        else {
                            (a, b, None)
                        };
                        result
                    }
                    "b" => {
                        let result = if b == 1 {
                            (a, b, Some(vals[1].parse::<isize>().unwrap()))
                        }
                        else {
                            (a, b, None)
                        };
                        result
                    }
                    _ => (0, 0, None)
                }
            },
            _ => (0, 0, None)

        };

        result
    }
}

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let instructions: Vec<Instruction> = buf_reader.lines().map(|line| {
            let line = line.unwrap();
            Instruction::new(&line)
        }
    ).collect();

    let mut index: isize = 0;
    let max_index = instructions.len() as isize;

    let mut a = 1;
    let mut b = 0;
    loop {
        println!("Index {index}, Insruction {:?}", instructions[index as usize]);
        // println!("      start of loop {a}, {b}");
        let (int_a, int_b, int_offset) = instructions[index as usize].interpret(a, b);

        // println!(  "{int_a}, {int_b}, {int_offset:?}");
        let offset = match int_offset {
            Some(int_offset) => int_offset,
            None => 1,
        };

        a = int_a;
        b = int_b;

        index += offset;
        if index >= max_index {
            break;
        }
    }

    println!("A: {a}, B: {b}");
}