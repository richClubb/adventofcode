use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
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
}

fn get_equal_bag_combinations(packages: &Vec<usize>, bag:&Bag, bag_list: &mut Vec<(Bag, Bag)>) {
    
    for index in 0..packages.len() {

        let mut packages_int = packages.clone();
        let next_package = packages_int.remove(index);

        let mut bag_1 = bag.clone();
        let bag_2 = Bag {packages: packages_int.clone()};

        bag_1.add_package(&next_package);

        let bag_1_weight = bag_1.clone().weight();
        let bag_2_weight = bag_2.weight();

        if bag_1_weight > bag_2_weight {
            return;
        }

        if bag_1_weight == bag_2_weight {
            bag_list.push((bag_1.clone(), bag_2.clone()));
        }

        get_equal_bag_combinations(&packages_int, &bag_1, bag_list);

    }
}

fn get_bag_combinations(packages: &Vec<usize>, bag: &Bag, smallest_bal_bag: &mut usize, smallest_bal_bag_qe: &mut usize, counter: &mut usize){

    for index in 0..packages.len() {
        
        let mut packages_int = packages.clone();
        let next_package = packages_int.remove(index);

        let mut bag_1 = bag.clone();

        bag_1.add_package(&next_package);
        let bag_2 = Bag {packages: packages_int.clone()};


        if bag_1.weight() * 2 > bag_2.weight() {
            return;
        }

        if (bag_1.weight() * 2) == bag_2.weight() {

            println!("Equals {bag_1:?} {bag_2:?}");

            let mut combos = Vec::<(Bag, Bag)>::new();
            println!("Getting combinations ");
            get_equal_bag_combinations(&bag_2.packages, &Bag{packages: Vec::<usize>::new()}, &mut combos);
            
            if combos.len() > 1 {

                if &bag_1.packages.len() < smallest_bal_bag {
                    println!("Smallest ");
                    *smallest_bal_bag = bag_1.packages.len();
                    *smallest_bal_bag_qe = bag_1.calc_qe();
                    println!("Smallest {smallest_bal_bag} {smallest_bal_bag_qe} {bag_1:?} {bag_2:?}")
                }
            }

            return;
        }

        if &bag_1.packages.len() <= smallest_bal_bag {
        // yoda typing
            if 2 < packages_int.len() {
                get_bag_combinations(&packages_int, &bag_1, smallest_bal_bag, smallest_bal_bag_qe, counter);
            }
        }
    }

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
    packages.reverse();

    println!("{packages:?}");

    let mut smallest_bag = std::usize::MAX;
    let mut qe_smallest_bag = std::usize::MAX;
    let mut counter = 0;
    
    get_bag_combinations(&packages, &Bag {packages: Vec::<usize>::new()}, &mut smallest_bag, &mut qe_smallest_bag, &mut counter);

    println!("{smallest_bag} {qe_smallest_bag}");

}