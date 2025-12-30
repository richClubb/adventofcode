use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct Person {
    name: String,
    relationships: HashMap<String, isize>,
}

fn possible_seating_combinations(remaining_people: &Vec<String>, curr_list: &Vec<String>, possible_combinations: &mut Vec<Vec<String>>) {
    
    for (index, _) in remaining_people.iter().enumerate() {
        let mut remaining_people = remaining_people.clone();
        let curr_person = remaining_people.remove(index);
        let mut curr_list = curr_list.clone();
        curr_list.push(curr_person);

        if remaining_people.len() == 0 {
            possible_combinations.push(curr_list.clone());
            return;
        }

        possible_seating_combinations(&remaining_people, &curr_list, possible_combinations);
    }

}

pub fn part_a(path: &String)
{
    println!("Part A & B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut people: HashMap<String, Person> = HashMap::new();
    let mut people_set: HashSet<String> = HashSet::new();

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
        people_set.insert(String::from(name1));
        people_set.insert(String::from(name2));

        let entry = people.entry(name1.to_string()).or_insert(Person {name: name1.to_string(), relationships: HashMap::new()});
        entry.relationships.insert(name2.to_string(), points);
    }

    let mut person_list: Vec<String> = Vec::new();

    for person in people_set {
        person_list.push(person.to_string());
    }

    person_list.sort();

    let mut possible_conbinations: Vec<Vec<String>> = Vec::new();

    let mut curr_list: Vec<String> = Vec::new();
    let base_person = person_list[0].clone();
    curr_list.push(base_person.clone());
    person_list.remove(0); 

    possible_seating_combinations(&person_list, &mut curr_list, &mut possible_conbinations);

    for (index, entry) in possible_conbinations.clone().iter().enumerate() {
        possible_conbinations[index].push(base_person.clone());
    }

    let mut max = std::isize::MIN;

    for combination in possible_conbinations {

        let mut total: isize = 0;

        for index in 0..(combination.len() - 1) {
            let person_1_name = combination[index].to_string();
            let person_2_name = combination[index + 1].to_string();

            let person = people.get(&person_1_name).unwrap();
            let relationship = person.relationships.get(&person_2_name).unwrap();
            total += relationship;

            let person = people.get(&person_2_name).unwrap();
            let relationship = person.relationships.get(&person_1_name).unwrap();
            total += relationship;
        }

        if total > max {
            max = total;
        }

    }

    println!("Total {:?}", max);

    
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_possible_seating_combinations() {
        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 1);
        }

        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice"), String::from("Bob")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 2);
        }

        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice"), String::from("Bob"), String::from("Charlie")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 6);
        }

        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice"), String::from("Bob"), String::from("Charlie"), String::from("Doug")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 24);
        }
        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice"), String::from("Bob"), String::from("Charlie"), String::from("Doug"), String::from("Evan")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 120);
        }

        {
            let mut combinations: Vec<Vec<String>> = Vec::new();
            let people: Vec<String> = Vec::from([String::from("Alice"), String::from("Bob"), String::from("Charlie"), String::from("Doug"), String::from("Evan"), String::from("Frank")]);
            let mut curr_list: Vec<String> = Vec::new(); 

            possible_seating_combinations(&people, &mut curr_list, &mut combinations);

            assert_eq!(combinations.len(), 720);
        }
    }
}