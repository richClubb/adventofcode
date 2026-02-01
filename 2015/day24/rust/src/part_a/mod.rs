use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::{HashSet, HashMap};

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
    successes: &mut HashMap<Vec<usize>, HashSet<Vec<usize>>>,
    indent: &str
) -> Option<(usize, usize)>{

    let mut smallest_bag = std::usize::MAX;
    let mut smallest_bag_qe = std::usize::MAX;

    let mut success = false;
    // println!("{indent}Ways to get target {packages:?}, {curr_bag:?}");
    if skips.contains(&(packages.clone(), curr_bag.clone())) {
        // println!("  caught skip 1");
        return None;
    }

    for package_index in (0..packages.len()).rev() {
        let mut remaining_packages = packages.clone();
        let curr_package = remaining_packages.remove(package_index);
        
        if (target - curr_bag.weight()) > *target {
            continue;
        }

        let mut new_bag = curr_bag.clone();
        new_bag.push(curr_package);
        new_bag.sort();

        if skips.contains(&(remaining_packages.clone(), new_bag.clone())) {
            // println!("  caught skip 2");
            // return None;
            break;
        }
    
        if (&new_bag).weight() < *target {

            let indent = format!("{indent}  ");
            let result = ways_to_get_target(&remaining_packages, target, &new_bag, skips, successes, indent.as_str());
            // println!("  result: {result:?}");
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
        else if (&new_bag).weight() == *target {
            let size = new_bag.len();
            // println!("{indent}  Found {:?} {remaining_packages:?} {successes:?}", new_bag);
            let indent = format!("{indent}  ");

            if remaining_packages.len() == 0 {
                return Some((new_bag.len(), (&new_bag).calc_qe()));
            }

            skips.insert((packages.clone(), curr_bag.clone()));

            successes.entry(new_bag.clone()).
                and_modify(|entry| {
                        entry.insert(remaining_packages.clone());
                    }
                ).
                or_insert(HashSet::from([remaining_packages.clone()]));

            let result = ways_to_get_target(&remaining_packages, target, &Vec::new(), skips, successes, indent.as_str());

            if result.is_none() {
                return None;
            }

            let size = result.unwrap().0;
            let qe = result.unwrap().1;

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

            // println!("{indent}  {new_bag:?} {remaining_packages:?}");
            // println!("{indent}  result {result:?}");

            let size = new_bag.len();
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

            success = true;

            // // println!("  successes: {smallest_bag:?} {smallest_bag_qe:?}");
            // return Some((smallest_bag, smallest_bag_qe));
            continue;
            
        }
    }

    skips.insert((packages.clone(), curr_bag.clone()));

    
    if success {
        // println!("{indent}Finished: {packages:?}, {curr_bag:?} - SUCCESSFUL");
        return Some((smallest_bag, smallest_bag_qe));
    }

    // println!("{indent}Finished: {packages:?}, {curr_bag:?} - UNSUCCESSFUL");
    return None;
}

fn get_bag_combinations(
    packages: &Vec<usize>, 
    target: &usize,
){
    let mut successes = HashMap::<Vec<usize>, HashSet::<Vec<usize>>>::new();
    let mut failures = HashSet::<(Vec<usize>, Vec<usize>)>::new();
    let result = ways_to_get_target(packages, target, &Vec::new(), &mut failures, &mut successes, "");

    let result = match result {
        Some((size, qe)) => (size, qe),
        None => (std::usize::MAX, std::usize::MAX),
    };

    let mut smallest_bag = result.0;
    let mut smallest_bag_qe = result.1;

    println!("Result: {smallest_bag:?} {smallest_bag_qe:?}");
}

pub fn part_a(path: &String, bag_count: &usize)
{
    println!("Bag count: {bag_count}");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let packages: Vec<usize> = buf_reader.lines().map(|line| {
            let line = line.unwrap();
            line.parse::<usize>().unwrap()
        }
    ).collect();

    let total: usize = packages.iter().sum();

    let target = total / bag_count;  // part b

    println!("Target: {target}");

    get_bag_combinations(&packages, &target);

}