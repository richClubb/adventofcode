use std::fs::File;
use std::io::{BufRead, BufReader};

use regex::Regex;

use crate::sue_properties::SueProperties;

pub fn part_a(sue_list_path: &String, sue_info_path: &String)
{
    println!("Part A");

    let file: File = File::open(sue_list_path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let sue_list_re = Regex::new(r"Sue\s(?<number>[0-9]{1,})\:(?<items>[a-z0-9\:\s,]{1,}){1,}").unwrap();

    let sue_list: Vec<(usize, SueProperties)> = buf_reader.lines().map(|line| {
            let curr_sue = sue_list_re.captures(&line.unwrap()).map(|caps| {
                    let sue_number = caps.name("number").unwrap().as_str().parse::<usize>().unwrap();
                    let info = caps.name("items").unwrap().as_str();

                    (sue_number as usize, SueProperties::new(&String::from(info)))
                }
            ).unwrap();

            curr_sue
        }
    ).collect();

    let sue_info_re = Regex::new(r"(?<item>[a-z]{1,})\:\s(?<qty>[0-9]{1,})").unwrap();

    let file: File = File::open(sue_info_path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let sue_info_vec: Vec<(String, usize)> = buf_reader.lines().map(|line| {
            sue_info_re.captures(&line.unwrap()).map(|caps | {
                    let item = caps.name("item").unwrap().as_str();
                    let quantity = caps.name("qty").unwrap().as_str().parse::<usize>().unwrap();

                    (item.to_string(), quantity)
                }
            ).unwrap()
        }
    ).collect();

    let mut ref_sue_info = SueProperties { 
        children: None, 
        cats: None, 
        samoyeds: None, 
        pomeranians: None, 
        akitas: None, 
        vizslas: None, 
        goldfish: None, 
        trees: None, 
        cars: None, 
        perfumes: None 
    };

    for (item, qty) in sue_info_vec {
        match item.as_str() {
            "children" => ref_sue_info.children = Some(qty), 
            "cats" => ref_sue_info.cats = Some(qty),
            "samoyeds" => ref_sue_info.samoyeds = Some(qty),
            "pomeranians" => ref_sue_info.pomeranians = Some(qty),
            "akitas" => ref_sue_info.akitas = Some(qty),
            "vizslas" => ref_sue_info.vizslas = Some(qty),
            "goldfish" => ref_sue_info.goldfish = Some(qty),
            "trees" => ref_sue_info.trees = Some(qty),
            "cars" => ref_sue_info.cars = Some(qty),
            "perfumes" => ref_sue_info.perfumes = Some(qty),
            _ => (),
        }
    }

    let total_filter = sue_list.iter().filter(|item| (item.1.children == ref_sue_info.children) || item.1.children.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.cats == ref_sue_info.cats) || item.1.cats.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.samoyeds == ref_sue_info.samoyeds) || item.1.samoyeds.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.pomeranians == ref_sue_info.pomeranians) || item.1.pomeranians.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.akitas == ref_sue_info.akitas) || item.1.akitas.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.vizslas == ref_sue_info.vizslas) || item.1.vizslas.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.goldfish == ref_sue_info.goldfish) || item.1.goldfish.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.trees == ref_sue_info.trees) || item.1.trees.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.cars == ref_sue_info.cars) || item.1.cars.is_none() );
    let total_filter = total_filter.filter(|item| (item.1.perfumes == ref_sue_info.perfumes) || item.1.perfumes.is_none() );

    let results: Vec<&(usize, SueProperties)> = total_filter.collect();

    println!("{:?}", results);
}