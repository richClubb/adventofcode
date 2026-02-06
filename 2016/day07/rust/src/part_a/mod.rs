use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

fn contains_abba(input: &String) -> bool {


    return true
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let line_re = Regex::new(r"(?<prefix>[a-z]{1,})\[(?<brackets>[a-z]{1,})\](?<suffix>[a-z]{1,})").unwrap();

    let result: Vec<bool> = buf_reader.lines().map(|line| {
            let line = line.unwrap();
            line_re.captures(&line).map(|caps| {
                    let prefix = caps.name("prefix").unwrap().as_str().to_string();
                    let brackets = caps.name("brackets").unwrap().as_str().to_string();
                    let suffix = caps.name("suffix").unwrap().as_str().to_string();

                    let prefix_result = contains_abba(&prefix);
                    let brackets_result = contains_abba(&brackets);
                    let suffix_result = contains_abba(&suffix);

                    let result = if (!brackets_result) && 
                    (prefix_result || suffix_result) {
                        true
                    } else {
                        false
                    };

                    result
                }
            ).unwrap()
        }
    ).filter(|entry| entry == &true).collect();

    println!("{}", result.len());
}