use std::fs::File;
use std::io::{BufRead, BufReader};

fn fit_containers(containers: Vec<usize>, curr_list: &Vec<usize>, target: usize, result: &mut Vec<Vec<usize>>) {

    let curr_total: usize = curr_list.iter().sum();

    if curr_total == target {
        result.push(curr_list.clone());
        return;
    }
    else if curr_total > target {
        return;
    }

    if containers.len() == 0 {
        return;
    }

    let mut new_containers = containers.clone();
    loop {
        let curr_container = new_containers.remove(0);

        let mut new_curr_list = curr_list.clone();
        new_curr_list.push(curr_container);

        fit_containers(new_containers.clone(), &new_curr_list, target, result);

        if new_containers.len() == 0 {
            return;
        }
    }

}

pub fn part_a(path: &String, size: &usize)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut containers: Vec<usize> = buf_reader.lines().map(|line| line.unwrap().parse::<usize>().unwrap()).collect();

    containers.sort();
    containers.reverse();

    let mut results: Vec<Vec<usize>> = Vec::new();
    let curr_list: Vec<usize> = Vec::new();

    fit_containers(containers, &curr_list, size.clone(), &mut results);

    println!("{:?}", results.len());
}