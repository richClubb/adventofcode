use std::fs::File;
use std::io::{BufRead, BufReader};

fn find_abas(input: &Vec<String>) -> Option<Vec<String>> {

    let mut abas = Vec::<String>::new();

    for entry in input {
        for index in 1..(entry.len() - 1) {
            let before = entry.chars().nth(index - 1).unwrap();
            let curr = entry.chars().nth(index).unwrap();
            let after = entry.chars().nth(index + 1).unwrap();
            if before == after {
                abas.push(format!("{before}{curr}{after}").to_string());
            }
        }
    }

    if abas.len() == 0 {
        return None;
    }

    return Some(abas)
}


pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let result: Vec<bool> = buf_reader.lines().map(|line| {
            let line = line.unwrap();

            let mut buffer = String::new();
            let mut supertext_sequences = Vec::<String>::new();
            let mut hypertext_sequences = Vec::<String>::new();

            for char in line.chars() {
                if char == '[' {
                    supertext_sequences.push(buffer.clone());
                    buffer.clear();
                    continue;
                }

                if char == ']' {
                    hypertext_sequences.push(buffer.clone());
                    buffer.clear();
                    continue;
                }

                buffer.push(char);
            }
            supertext_sequences.push(buffer.clone());

            // println!("{supertext_sequences:?} {hypertext_sequences:?}");

            let abas = find_abas(&supertext_sequences);
            let babs = find_abas(&hypertext_sequences);
            // println!("{abas:?}");

            let result = if abas.is_some() && babs.is_some() {

                let abas = abas.unwrap();
                let babs = babs.unwrap();
                let mut success = false;

                for entry in abas {
                    let char_1 = entry.chars().nth(0).unwrap();
                    let char_2 = entry.chars().nth(1).unwrap();
                    let bab = format!("{char_2}{char_1}{char_2}");
                    if babs.contains(&bab) {
                        success = true;
                        break;
                    }
                }

                success
            }
            else {
                false
            };

            // println!("{line} {result}");
            result
        }
    ).filter(|entry| entry == &true).collect();

    println!("{}", result.len());
}