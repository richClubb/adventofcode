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

fn get_bag_combinations(packages: &Vec<usize>, target_weight: &usize, bags: &mut (Option<Bag>, Option<Bag>, Option<Bag>),  bag_list: &mut Vec<(Bag, Bag, Bag)>) {
    
    for package_index in 0..packages.len() {
        let mut packages_int = packages.clone();

        let curr_package = packages_int.remove(package_index);

        if bags.0.is_none() {
            bags.0 = Some(Bag{packages: Vec::from([curr_package.clone()])});
        }

        if bags.0.is_some() && bags.1.is_none() {

        }
        
        let bag_0_weight = bags.0.as_ref().unwrap().weight();
        if &bag_0_weight > target_weight {
            return;
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

    // whatever you do, do it greedily
    packages.reverse();

    let total: usize = packages.iter().sum();

}