use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone)]
struct Bag {
    packages: Vec<usize>
}

impl Bag {
    pub fn weight(&self) -> usize {
        return self.packages.iter().sum();
    }

    pub fn add_package(&mut self, package: &usize) {
        self.packages.push(*package);
    }
}

fn get_bag_combinations(packages: &Vec<usize>) -> (Vec<Bag>, Vec<Bag>, Vec<Bag>) {

    let mut packages_mut = packages.clone();

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

    let mut bag_1 = Vec::<usize>::new();
    let mut bag_2 = Vec::<usize>::new();
    let mut bag_3 = Vec::<usize>::new();

    println!("{packages:?}");


}