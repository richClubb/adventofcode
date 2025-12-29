use std::fs::File;
use std::io::{BufRead, BufReader, Lines};

fn contains_two_pairs(input: &String) -> bool {

    for index in 0..(input.len() - 1) {

        let match_str = format!("{}{}", &input.chars().nth(index).unwrap(), &input.chars().nth(index + 1).unwrap());
        
        // guaranteed to have at least one match
        let last_match = input.rfind(&match_str).unwrap() as i16;

        if last_match >= index as i16 + 2 {
            return true;
        }
    }

    return false;
}

fn one_letter_repeat_with_space(input: &String) -> bool {

    for index in 0..(input.len() - 2) {

        let start_char = input.chars().nth(index).unwrap();        
        // guaranteed to have at least one match
        let match_char = input.chars().nth(index + 2).unwrap();

        if start_char == match_char {
            return true;
        }
    }

    return false;

}

fn is_good_word(input: &String) -> bool {
    if contains_two_pairs(input) && one_letter_repeat_with_space(input) {
        println!("{} is good", input);
        return true;
    }

    println!("{} is bad", input);
    return false;
}

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let results: Vec<Result<String, std::io::Error>> = buf_reader.lines().filter(|line| is_good_word(&line.as_ref().unwrap())).collect();
    
    println!("{}", results.len());

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_two_pairs() {
        assert_eq!(contains_two_pairs(&String::from("abc")), false);
        assert_eq!(contains_two_pairs(&String::from("aaa")), false);
        assert_eq!(contains_two_pairs(&String::from("aaaa")), true);
        assert_eq!(contains_two_pairs(&String::from("abab")), true);
        assert_eq!(contains_two_pairs(&String::from("aaaa")), true);
        assert_eq!(contains_two_pairs(&String::from("afaf")), true);
        assert_eq!(contains_two_pairs(&String::from("abcdab")), true);
        assert_eq!(contains_two_pairs(&String::from("xxyxx")), true);
    }

    #[test]
    fn test_one_letter_repeat_with_space() {
        assert_eq!(one_letter_repeat_with_space(&String::from("abc")), false);
        assert_eq!(one_letter_repeat_with_space(&String::from("aba")), true);
        assert_eq!(one_letter_repeat_with_space(&String::from("abcbd")), true);
        assert_eq!(one_letter_repeat_with_space(&String::from("abcad")), false);
        assert_eq!(one_letter_repeat_with_space(&String::from("xxyxx")), true);
    }

    #[test]
    fn test_is_good() {
        assert_eq!(is_good_word(&String::from("abc")), false);
        assert_eq!(is_good_word(&String::from("abcabdbc")), true);
        assert_eq!(is_good_word(&String::from("xxyxx")), true);
    }

}