use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use std::collections::HashSet;

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut replacements: Vec<(String, String)> = Vec::<(String, String)>::new();
    let replacement_re = Regex::new(r"(?<origin>[A-Za-z]{1,})\s=>\s(?<replacement>[A-Za-z]{1,})").unwrap();
    let mut base_string = String::new();

    for line in buf_reader.lines() {
        let line = line.unwrap();
        let result = replacement_re.captures(&line).map(|caps| {
                let origin = caps.name("origin").unwrap().as_str();
                let replacement = caps.name("replacement").unwrap().as_str();
                return (origin.to_string(), replacement.to_string());
            }
        );

        if result.is_some() {
            let (origin, replacement) = result.unwrap();
            replacements.push((origin, replacement));
        }
        else {
            if &line.len() == &0 {
                continue
            }
            else {
                base_string = line.to_string();
            }
        }
    }

    replacements.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let mut count: usize = 0;
    let mut new_base_string = base_string.clone();
    println!("Start: {new_base_string}");
    loop {
        let mut match_performed = false;
        for (origin, replacement) in replacements.clone() {
            if new_base_string.contains(&replacement) {
                count += 1;
                new_base_string = new_base_string.replacen(&replacement, &origin, 1);
                println!("new_base: {new_base_string}");
                match_performed = true;
            }
        }

        if match_performed == false {
            println!("{new_base_string}");
            break;
        }
    }

    println!("{count}")

}