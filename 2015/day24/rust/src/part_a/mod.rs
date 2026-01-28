use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::{HashSet, HashMap};

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct Bag {
    packages: Vec<usize>
}

impl Bag {
    pub fn weight(&self) -> usize {
        return self.packages.iter().sum();
    }

    pub fn add_package(&mut self, package: &usize) {
        self.packages.push(*package);
        self.packages.sort();
    }

    pub fn calc_qe(&self) -> usize {
        return self.packages.iter().fold(1 as usize, |acc, entry| {
                let (result, of_flag) = acc.overflowing_mul(*entry);
                if of_flag {
                    std::usize::MAX
                }
                else {
                    result
                }
            }
        );
    }

    pub fn size(&self) -> usize {
        return self.packages.len();
    }
}

fn get_all_splits(packages: &Vec<usize>, target: &usize, bag: &Bag, fails: &mut HashSet<(Bag, Vec<usize>)>, pairs: &mut HashSet<Vec<Bag>>) {

    // println!("Get all splits {bag:?} {packages:?}");

    if fails.contains(&(bag.clone(), packages.clone())) {
        // println!("  Caught fail in get all splits {bag:?} {packages:?}");
        return;
    }

    for package_index in 0..packages.len() {

        
        let mut packages = packages.clone();
        let curr_package = packages.remove(package_index);
        
        // println!("  curr package: {curr_package}");

        let remaining_weight = target - bag.weight();
        if curr_package > remaining_weight {
            // println!("    already too heavy");
            continue;
        }

        // println!("  {curr_package} {packages:?}");

        if &(bag.weight() + curr_package) < target {
            let mut new_bag = bag.clone();
            new_bag.add_package(&curr_package);
            // println!("  under weight");

            if fails.contains(&(bag.clone(), packages.clone())) {
                return;
            }

            get_all_splits(&packages, target, &new_bag, fails, pairs);
        }

        if &(bag.weight() + curr_package) == target {
            // println!("  right weight");
            let mut new_bag_1 = bag.clone();
            new_bag_1.add_package(&curr_package);
            let mut packages = packages.clone();
            packages.sort();
            let new_bag_2 = Bag {packages: packages.clone()};
            let mut bags = Vec::from([new_bag_1, new_bag_2]);
            bags.sort_by(|a, b| b.size().cmp(&a.size()));

            pairs.insert(bags);

            return;
        }

        if &(bag.weight() + curr_package) > target {
            // println!("  over weight");
            continue;
        }
    }

    // println!("Unsuccessful get all packages {bag:?} {packages:?}");
    fails.insert((bag.clone(), packages.clone()));
}

fn get_bag_combinations(packages: &Vec<usize>, target: &usize, curr_bag: &Bag, fails: &mut HashSet<(Bag, Vec<usize>)>, successes: &mut HashMap<Bag, HashSet<Vec<Bag>>>) {

    if successes.contains_key(curr_bag) {
        // println!("  Caught success in get_bag_combinations {curr_bag:?} {packages:?}");
        return;
    }

    if fails.contains(&(curr_bag.clone(), packages.clone())) {
        // println!("  Caught fail in get_bag_combinations {curr_bag:?} {packages:?}");
        return;
    }

    for package_index in 0..packages.len() {

        let mut packages = packages.clone();
        let curr_package = packages.remove(package_index);

        if &(curr_bag.weight() + curr_package) < target {
            let mut new_bag = curr_bag.clone();
            new_bag.add_package(&curr_package);

            get_bag_combinations(&packages, target, &new_bag, fails, successes);
        }

        if &(curr_bag.weight() + curr_package) == target {
            let mut new_bag = curr_bag.clone();
            new_bag.add_package(&curr_package);

            if successes.contains_key(&new_bag) {
                return;
            }

            let mut pairs = HashSet::<Vec<Bag>>::new();
            
            println!("Getting splits for bag: {new_bag:?}");
            get_all_splits(&packages, target, &Bag{ packages: Vec::<usize>::new()}, fails, &mut pairs);
            println!("Got splits for {new_bag:?} {pairs:?}");
            successes.insert(new_bag, pairs);
            return;
        }

        if &(curr_bag.weight() + curr_package) > target {
            return;
        }
    }

    // println!("unsuccessful get bag combinations {curr_bag:?} {packages:?}");
    fails.insert((curr_bag.clone(), packages.clone()));
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let packages: Vec<usize> = buf_reader.lines().map(|line| {
            let line = line.unwrap();
            line.parse::<usize>().unwrap()
        }
    ).collect();

    let mut packages = packages.clone();

    // whatever you do, do it greedily
    packages.reverse();

    let total: usize = packages.iter().sum();
    let target = total / 3;

    println!("{packages:?}");

    let mut successes = HashMap::<Bag, HashSet<Vec<Bag>>>::new();
    let mut fails = HashSet::<(Bag, Vec<usize>)>::new();

    println!("Target: {target}");

    get_bag_combinations(&packages, &target, &mut Bag{ packages: Vec::<usize>::new() }, &mut fails, &mut successes);

    let mut smallest_bag = std::usize::MAX;
    let mut smallest_bag_qe = std::usize::MAX;
    for (bag, others) in successes {

        if (bag.size() <= smallest_bag) && (bag.calc_qe() < smallest_bag_qe) {
            smallest_bag = bag.size();
            smallest_bag_qe = bag.calc_qe();
        }

        for other in others {

            for bag in other {
                if (bag.size() <= smallest_bag) && (bag.calc_qe() < smallest_bag_qe) {
                    smallest_bag = bag.size();
                    smallest_bag_qe = bag.calc_qe();
                }
            }
        }
    }

    println!("Result: {smallest_bag} {smallest_bag_qe}");
}