use std::fs::File;
use std::io::{BufRead, BufReader, Lines};

fn contains_bad_strings(input: &String) -> bool
{
    if input.contains("ab") || input.contains("cd") || input.contains("pq") || input.contains("xy") {
        return true;
    }

    return false;
}

fn contains_double_letters(input: &String) -> bool
{
    for character in 'a'..='z' {
        let match_string = format!("{0}{0}", character);

        if input.contains(&match_string)
        {
            return true;
        }
    }
    
    return false;
}

fn is_vowel(character: &char) -> bool {
    if (character == &'a') || 
    (character == &'e') ||
    (character == &'i') ||
    (character == &'o') ||
    (character == &'u') {
        return true;
    }

    return false;
}

fn contains_three_vowels(input: &String) -> bool
{
    let results: Vec<char> = input.chars().into_iter().filter(|character| is_vowel(character)).collect();

    if results.len() >= 3 {
        return true;
    }

    return false;
}

fn is_good_word(input: &String) -> bool
{
    let cbs = contains_bad_strings(input);
    let ctv = contains_three_vowels(input);
    let cdl = contains_double_letters(input);

    if !cbs && ctv && cdl {
        return true;
    }

    return false;
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let results: Vec<Result<String, std::io::Error>> = buf_reader.lines().filter(|line| is_good_word(&line.as_ref().unwrap())).collect();

    println!("{}", results.len());

    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bad_strings() {
        assert_eq!(contains_bad_strings(&String::from("abc")), true);
        assert_eq!(contains_bad_strings(&String::from("aaa")), false);
        assert_eq!(contains_bad_strings(&String::from("cdd")), true);
        assert_eq!(contains_bad_strings(&String::from("cee")), false);
        assert_eq!(contains_bad_strings(&String::from("pqr")), true);
        assert_eq!(contains_bad_strings(&String::from("prr")), false);
        assert_eq!(contains_bad_strings(&String::from("qrr")), false);
        assert_eq!(contains_bad_strings(&String::from("xyz")), true);
        assert_eq!(contains_bad_strings(&String::from("xzy")), false);
    }

    #[test]
    fn test_3_vowels() {
        assert_eq!(contains_three_vowels(&String::from("aaa")), true);
        assert_eq!(contains_three_vowels(&String::from("aae")), true);
        assert_eq!(contains_three_vowels(&String::from("aei")), true);
        assert_eq!(contains_three_vowels(&String::from("aeio")), true);
        assert_eq!(contains_three_vowels(&String::from("aeiou")), true);
        assert_eq!(contains_three_vowels(&String::from("aaaaei")), true);
        assert_eq!(contains_three_vowels(&String::from("abb")), false);

    }

    #[test]
    fn test_contains_duplicate_letters() {
        assert_eq!(contains_double_letters(&String::from("aaa")), true);
        assert_eq!(contains_double_letters(&String::from("abc")), false);
        assert_eq!(contains_double_letters(&String::from("aaadd")), true);
        assert_eq!(contains_double_letters(&String::from("aaeea")), true);
    }
}