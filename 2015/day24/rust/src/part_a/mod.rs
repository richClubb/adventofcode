use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::{HashSet, HashMap};

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
struct Bag {
    packages: Vec<usize>
}

pub trait Weight {
    fn weight(&self) -> usize;
}

impl Weight for &Vec<usize> {
    fn weight(&self) -> usize {
        return self.iter().sum();
    }
}

pub trait QeCalc {
    fn calc_qe(&self) -> usize;
}

impl QeCalc for &Vec<usize> {

    fn calc_qe(&self) -> usize {
        return self.iter().fold(1 as usize, |acc, entry| {
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
}

fn ways_to_get_target(
    packages: &Vec<usize>, 
    target: &usize, 
    curr_bag: &Vec<usize>, 
    skips: &mut HashSet<(Vec<usize>, Vec<usize>)>, 
    successes: &mut HashMap<Vec<usize>, HashSet<Vec<usize>>>) -> Option<(usize, usize)>{

    let mut smallest_bag = std::usize::MAX;
    let mut smallest_bag_qe = std::usize::MAX;

    let mut success = false;
    // println!("Ways to get target {packages:?}, {curr_bag:?}");
    if skips.contains(&(packages.clone(), curr_bag.clone())) {
        // println!("caught skip");
        return None;
    }

    for package_index in (0..packages.len()).rev() {
        let mut remaining_packages = packages.clone();
        let curr_package = remaining_packages.remove(package_index);

        if (target - curr_bag.weight()) > *target {
            continue;
        }

        if (curr_bag.weight() + curr_package) < *target {
            let mut new_bag = curr_bag.clone();
            new_bag.push(curr_package);
            new_bag.sort();

            if skips.contains(&(packages.clone(), curr_bag.clone())) {
                // println!("caught skip");
                return None;
            }

            let result = ways_to_get_target(&remaining_packages, target, &new_bag, skips, successes);
            //println!("  result: {result:?}");
            match result {
                Some((size, qe)) => {
                    if size < smallest_bag {
                        smallest_bag = size;
                        smallest_bag_qe = qe;
                    }
                    if size == smallest_bag {
                        if qe < smallest_bag_qe {
                            smallest_bag = size;
                            smallest_bag_qe = qe;
                        }
                    }
                    success = true;
                },
                None => (),
            }
        }
        else if (curr_bag.weight() + curr_package) == *target {
            let mut new_bag = curr_bag.clone();
            new_bag.push(curr_package);
            new_bag.sort();
            let size = new_bag.len();
            // println!("  Found {:?} {packages:?}", curr_bag);
            successes.entry(new_bag.clone()).
                and_modify(|entry| {
                        entry.insert(remaining_packages.clone());
                    }
                ).
                or_insert(HashSet::from([remaining_packages.clone()]));
            // println!("  adding to skips {packages:?}, {curr_bag:?}");
            skips.insert((packages.clone(), curr_bag.clone()));

            let qe = (&new_bag).calc_qe();
            
            if size < smallest_bag {
                smallest_bag = size;
                smallest_bag_qe = qe;
            }
            if size == smallest_bag {
                if qe < smallest_bag_qe {
                    smallest_bag = size;
                    smallest_bag_qe = qe;
                }
            }

            // println!("  successes: {smallest_bag:?} {smallest_bag_qe:?}");
            return Some((smallest_bag, smallest_bag_qe));
            
        }
    }

    // println!("Failed {packages:?} {curr_bag:?}");
    skips.insert((packages.clone(), curr_bag.clone()));

    if success {
        return Some((smallest_bag, smallest_bag_qe));
    }

    return None;
}

fn get_bag_combinations(
    packages: &Vec<usize>, 
    target: &usize,
){
    let mut successes = HashMap::<Vec<usize>, HashSet::<Vec<usize>>>::new();
    let mut failures = HashSet::<(Vec<usize>, Vec<usize>)>::new();
    let result = ways_to_get_target(packages, target, &Vec::new(), &mut failures, &mut successes);

    let result = match result {
        Some((size, qe)) => (size, qe),
        None => (std::usize::MAX, std::usize::MAX),
    };

    let mut smallest_bag = result.0;
    let mut smallest_bag_qe = result.1;

    println!("  {smallest_bag:?} {smallest_bag_qe:?}");

    let mut success_successes = HashMap::<Vec<usize>, HashSet::<Vec<usize>>>::new();
    let mut success_failures = HashSet::<(Vec<usize>, Vec<usize>)>::new();
    
    for success in successes {
        println!("{success:?}");
    
        let success_packages = success.1.iter().nth(0).unwrap();
        let result = ways_to_get_target(&success_packages, target, &Vec::new(), &mut success_failures, &mut success_successes);

        match result {
                Some((size, qe)) => {
                    if size < smallest_bag {
                        smallest_bag = size;
                        smallest_bag_qe = qe;
                    }
                    if size == smallest_bag {
                        if qe < smallest_bag_qe {
                            smallest_bag = size;
                            smallest_bag_qe = qe;
                        }
                    }
                },
                None => (),
            }

        println!("result {result:?}");
    }

    println!("  {smallest_bag:?} {smallest_bag_qe:?}");
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

    let total: usize = packages.iter().sum();
    // let target = total / 2;
    let target = total / 4;

    println!("{packages:?}");

    let mut successes = HashMap::<Bag, HashSet<Vec<Bag>>>::new();
    let mut fails = HashSet::<(Bag, Vec<usize>)>::new();

    println!("Target: {target}");

    get_bag_combinations(&packages, &target);

    let mut smallest_bag = std::usize::MAX;
    let mut smallest_bag_qe = std::usize::MAX;

    println!("Result: {smallest_bag} {smallest_bag_qe}");
}