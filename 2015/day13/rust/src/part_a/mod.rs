use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use std::collections::{HashMap, HashSet};

struct Person {
    name: String,
    relationships: HashMap<String, isize>,
}

struct Seating {
    person: String,
    left: String,
    left_score: isize,
    right: String,
    right_score: isize,
}

fn possible_seating_combinations(people: &HashSet<String>) {
    
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut people: HashSet<String> = HashSet::new();

    let line_re = Regex::new(r"(?<name1>[A-Za-z]{1,})\swould\s(?<intensity>gain|lose)\s(?<points>[0-9]{1,})\shappiness\sunits\sby\ssitting\snext\sto\s(?<name2>[A-Za-z]{1,})\.").unwrap();

    for line in buf_reader.lines() {

        let line = line.unwrap();
        let (name1, name2, points) = line_re.captures(&line).map(|caps| {
                let name1 = caps.name("name1").unwrap().as_str();
                let name2 = caps.name("name2").unwrap().as_str();
                let intensity = caps.name("intensity").unwrap().as_str();
                let points_scalar = caps.name("points").unwrap().as_str().parse::<isize>().unwrap();
                let points_vector = match intensity {
                    "gain" => points_scalar,
                    "lose" => -points_scalar,
                    _ => 0,
                };

                (name1, name2, points_vector)
            }
        ).unwrap();
        println!("{} {} {}", name1, name2, points);
        people.insert(String::from(name1));
        people.insert(String::from(name2));
    }

    println!("{:?}", people);

    
}