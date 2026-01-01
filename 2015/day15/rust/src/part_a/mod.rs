use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

#[derive(Debug)]
struct Ingredient {
    name: String,
    capacity: isize,
    durability: isize,
    flavour: isize,
    texture: isize,
    calories: usize
}

impl Ingredient {
    pub fn new(name: String, capacity: isize, durability: isize, flavour: isize, texture: isize, calories: usize) -> Self {
        return Ingredient { name, capacity, durability, flavour, texture, calories };
    }
}

fn calc_cookie(ingredients: &Vec<Ingredient>, quantity: Vec<usize>) -> usize {

    let mut capacity: isize = 0;
    for index in 0..ingredients.len() {
        capacity += ingredients[index].capacity * (quantity[index] as isize);
    }

    if capacity <= 0 {
        return 0;
    }

    let mut durability: isize = 0;
    for index in 0..ingredients.len() {
        durability += ingredients[index].durability * (quantity[index] as isize);
    }

    if durability <= 0 {
        return 0;
    }

    let mut flavour: isize = 0;
    for index in 0..ingredients.len() {
        flavour += ingredients[index].flavour * (quantity[index] as isize);
    }

    if flavour <= 0 {
        return 0;
    }

    let mut texture: isize = 0;
    for index in 0..ingredients.len() {
        texture += ingredients[index].texture * (quantity[index] as isize);
    }

    if texture <= 0 {
        return 0;
    }


    return (capacity as usize)*(durability as usize)*(flavour as usize)*(texture as usize);
}

fn calc_possible_combinations(base: &Vec<usize>, base_index: usize, max: usize, result: &mut Vec<Vec<usize>>) {

    if base_index == (base.len() - 1)  {
        for index in 1..=max {
            let mut new_base = base.clone();
            new_base[base_index] = index;
            if new_base.iter().sum::<usize>() == 100 {
                //println!("Adding {:?}", new_base);
                result.push(new_base);
            }
        }

        return;
    }

    for index in 1..=max {
        
        let mut new_base = base.clone();
        new_base[base_index] = index;
        //println!("recursive case {} {:?}", base_index, new_base);

        calc_possible_combinations(&new_base, base_index + 1, max, result);

    }

    
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let line_re = Regex::new(r"(?<ingredient>[A-z]{1,}[a-z]{1,})\:\scapacity\s(?<capacity>[-0-9]{1,})\,\sdurability\s(?<durability>[-0-9]{1,})\,\sflavor\s(?<flavor>[-0-9]{1,})\,\stexture\s(?<texture>[-0-9]{1,})\,\scalories\s(?<calories>[-0-9]{1,})").unwrap(); 

    let mut ingredients: Vec<Ingredient> = Vec::new();

    for line in buf_reader.lines() {
        let line = line.unwrap();

        let ingredient = line_re.captures(&line).map(|caps| 
            {
                let name = caps.name("ingredient").unwrap().as_str();
                let capacity = caps.name("capacity").unwrap().as_str().parse::<isize>().unwrap();
                let durability  = caps.name("durability").unwrap().as_str().parse::<isize>().unwrap();
                let flavour = caps.name("flavor").unwrap().as_str().parse::<isize>().unwrap();
                let texture = caps.name("texture").unwrap().as_str().parse::<isize>().unwrap();
                let calories = caps.name("calories").unwrap().as_str().parse::<usize>().unwrap();

                Ingredient::new(name.to_string(), capacity, durability, flavour, texture, calories)
            }
        ).unwrap();
    
        ingredients.push(ingredient);
        
    }

    let max_quantity = 101 -ingredients.len();

    let base: Vec<usize> = vec![1; ingredients.len()];
    let mut results: Vec<Vec<usize>> = Vec::new();
    calc_possible_combinations(&base, 0, max_quantity, &mut results);

    let mut max = std::usize::MIN;
    for result in results {
        let val = calc_cookie(&ingredients, result);
        if val > max {
            max = val;
        }
    }

    println!("Max {max}");

}