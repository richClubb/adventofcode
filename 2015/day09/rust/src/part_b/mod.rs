use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;
use std::collections::{HashSet, HashMap};

#[derive(Debug, Eq, Hash, PartialEq)]
struct Journey {
    origin: String,
    destination: String
}

fn calculate_route_distance(journey: &Vec<String>, journeys: &HashMap<Journey, usize>) -> usize {

    let mut start = journey.first().unwrap();
    let mut distance = 0;

    let slice = &journey[1..];
    for a in slice {
        let curr_leg = Journey{ origin:start.to_string(), destination: a.to_string()};
        distance += journeys[&curr_leg];
        start = a;
    }

    return distance;
}

fn calculate_all_possible_routes(start: &String, journey: &mut Vec<String>, cities: &HashSet<String>, result: &mut Vec<Vec<String>>) {

    journey.push(start.to_string());

    let mut remaining_cities = cities.clone();
    remaining_cities.remove(start);

    if remaining_cities.len() == 0 {
        let mut reversed = journey.clone();
        reversed.reverse();
        if !result.contains(&reversed) {
            result.push(journey.clone());
        }
    }

    for city in &remaining_cities {
        let mut curr_journey = journey.clone();
        calculate_all_possible_routes(city, &mut curr_journey, &remaining_cities, result);
    }
}

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut cities: HashSet<String> = HashSet::new();
    let mut journeys: HashMap<Journey, usize> = HashMap::new();

    for line in buf_reader.lines() {
        
        let line = line.unwrap();
        let re = Regex::new(r"(?<origin>[a-zA-Z]{1,})\sto\s(?<destination>[a-zA-Z]{1,})\s=\s(?<distance>[0-9]{1,})").unwrap();

        let (origin, destination, distance) = re.captures(&line).map(|caps| {
                let origin = caps.name("origin").unwrap().as_str();
                let destination = caps.name("destination").unwrap().as_str();
                let distance = caps.name("distance").unwrap().as_str().parse::<usize>().unwrap();

                (origin, destination, distance)
            }
        ).unwrap();

        cities.insert(origin.to_string());
        cities.insert(destination.to_string());

        let o_to_d = Journey{ origin: origin.to_string(), destination: destination.to_string()};
        let d_to_o = Journey{ origin: destination.to_string(), destination: origin.to_string()};

        journeys.insert(o_to_d, distance);
        journeys.insert(d_to_o, distance);
    }

    println!("Possible cities {:?}", cities);

    let mut results: Vec<Vec<String>> = Vec::new();
    for city in &cities {
        let mut journey: Vec<String> = Vec::new();
        calculate_all_possible_routes(city, &mut journey, &cities, &mut results);
    }
    println!("{:?}", results.len());

    let mut max_distance = std::usize::MIN;
    for result in results {
        let result = calculate_route_distance(&result, &journeys);
        if result > max_distance {
            max_distance = result;
        }
    }

    println!("{max_distance}");

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_distance() {

 

    }
}