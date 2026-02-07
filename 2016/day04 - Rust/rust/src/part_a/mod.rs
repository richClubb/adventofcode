use std::fs::File;
use std::io::{BufRead, BufReader};

use std::collections::HashMap;

use regex::Regex;



fn decrypt(ciphertext: &String, key: &usize) -> String {

    let mut plaintext = String::new();
    for character in ciphertext.chars()
    {
        if (character >= 'a') && (character <= 'z')
        {
            let mut base: u8 = character as u8 - 97;
            base = ((base as usize + *key) % 26) as u8;
            let base = base + 97;
            let new_char = base as char;
            plaintext.push(new_char);
        }
        else if character == '-'
        {
            plaintext.push(' ');
        }
    }
    return plaintext;
}


pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let line_re = Regex::new(r"(?<code>[a-z\-]{1,})\-(?<sector_id>[0-9]{1,})\[(?<checksum>[a-z]{1,})\]").unwrap();

    let mut sum_of_sectors = 0;
    let results: Vec<(String, usize, String, bool, String)> = buf_reader.lines().map(|line| {
            let line = line.unwrap();
            line_re.captures(&line).map(|caps| {
                    let code = caps.name("code").unwrap().as_str().to_string();
                    let sector_id = caps.name("sector_id").unwrap().as_str().parse::<usize>().unwrap();
                    let checksum = caps.name("checksum").unwrap().as_str().to_string();
                    
                    let mut char_count = HashMap::<usize, Vec<char>>::new();

                    for character in 'a'..='z' {
                        let num_of_character = code.chars().fold(std::usize::MIN,|acc, entry| { 
                                if character == entry {
                                    acc + 1
                                }
                                else {
                                    acc + 0
                                }
                            }
                        );
                        char_count.entry(num_of_character).and_modify(|entry| {
                                entry.push(character);
                                entry.sort();
                            }
                        ).or_insert(Vec::<char>::from([character]));
                    }

                    let mut keys: Vec<&usize> = char_count.keys().into_iter().collect();
                    keys.sort();

                    let mut chars = Vec::<char>::new();

                    for key in keys.iter().rev(){
                        let temp = &char_count[key];
                        chars.extend(temp);
                    }

                    let mut checksum_check = String::new();
                    for index in 0..5 {
                        checksum_check.push(chars[index]);
                    }

                    if checksum_check.eq(&checksum) {
                        sum_of_sectors += sector_id;
                    }

                    let plaintext = decrypt(&code, &sector_id);

                    (code, sector_id, checksum.clone(), checksum_check.eq(&checksum), plaintext)
                }
            ).unwrap()
        }
    ).filter(|(_, _, _, checksum_match, _)| {
            checksum_match == &true
        }
    ).collect();

    println!("{}", results.len());

    for result in results{
        println!("{}: {}", result.4, result.1);
    }

    println!("sum of sectors: {sum_of_sectors}");
}