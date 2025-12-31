use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use std::collections::HashMap;

#[derive(Debug, Clone)]
struct ReindeerStats {
    name: String,
    active_speed: usize,
    active_time: usize,
    rest_time: usize
}

impl ReindeerStats {

    pub fn new(name: String, active_speed: usize, active_time: usize, rest_time: usize) -> Self {
        return ReindeerStats { name, active_speed, active_time, rest_time };
    }

    pub fn run(&self, time: usize) -> usize {
        let total_runtime = self.active_time + self.rest_time;
        let iterations = time / total_runtime;

        let distance = if (time as isize - (iterations * total_runtime + self.active_time) as isize) >= 0 {
            self.active_speed * self.active_time * (iterations + 1)
        }
        else {
            let leftover_time = time - (iterations * total_runtime);
            (self.active_speed * self.active_time * iterations) + (self.active_speed * leftover_time)
        };

        return distance;
    }

}

pub fn part_b(path: &String, length: usize)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let line_re = Regex::new(r"(?<name>[A-Za-z]{1,})\scan\sfly\s(?<speed>[0-9]{1,})\skm\/s\sfor\s(?<active_time>[0-9]{1,})\sseconds\,\sbut\sthen\smust\srest\sfor\s(?<rest_time>[0-9]{1,})\sseconds\.").unwrap();

    let mut reindeer: HashMap<String, ReindeerStats> = HashMap::new();
    let mut score: HashMap<String, usize> = HashMap::new();

    for line in buf_reader.lines() {

        let line = line.unwrap();
        let (name, speed, active_time, rest_time) = line_re.captures(&line).map(|caps| {
                let name = caps.name("name").unwrap().as_str();
                let speed = caps.name("speed").unwrap().as_str().parse::<usize>().unwrap();
                let active_time = caps.name("active_time").unwrap().as_str().parse::<usize>().unwrap();
                let rest_time = caps.name("rest_time").unwrap().as_str().parse::<usize>().unwrap();

                (name.to_string(), speed, active_time, rest_time)
            }
        ).unwrap();

        reindeer.insert(name.clone(), ReindeerStats::new(name.clone(), speed, active_time, rest_time) );
        score.insert(name.clone(), 0);
    }

    for second in 1..length {

        let mut furthest_distance = 0;
        let mut distances: HashMap<String, usize> = HashMap::new();

        for (name, deer) in &reindeer {
            let distance = deer.run(second);
            distances.insert(name.to_string(), distance);

            if distance > furthest_distance {
                furthest_distance = distance;
            }
        }

        for (name, distance) in &distances {
            if distance == &furthest_distance {
                *score.entry(name.to_string()).or_insert(0) += 1;
            }
        }
    }

    println!("{:?}", score);
}