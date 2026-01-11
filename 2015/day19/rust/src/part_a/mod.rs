use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use std::collections::HashSet;

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut replacements: Vec<(String, String)> = Vec::<(String, String)>::new();
    let replacement_re = Regex::new(r"(?<origin>[A-Za-z]{1,})\s=>\s(?<replacement>[A-Za-z]{1,})").unwrap();
    let mut base_string = String::new();

    let mut new_mixes = HashSet::<String>::new();    

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

    for (origin, replacement) in replacements {
        let thing: Vec<(usize, &str)> = base_string.match_indices(&origin).collect();
        thing.iter().for_each(|item| {
                let str1 = base_string[0..item.0].to_string();
                let str2 = base_string[item.0 + origin.len()..].to_string();
                let new_str = str1 + &replacement + &str2;
                new_mixes.insert(new_str);
            }
        );
    }

    println!("{}", new_mixes.len());

}