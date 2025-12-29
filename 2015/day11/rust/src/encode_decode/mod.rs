use std::collections::HashMap;

pub fn encode_string(input: &String) -> Vec<u8> {

    let alphabet : HashMap<char, u8> = HashMap::from(
        [
            ('a',  0), ('b',  1), ('c',  2), ('d',  3), ('e',  4), ('f',  5), 
            ('g',  6), ('h',  7), ('i',  8), ('j',  9), ('k', 10), ('l', 11),
            ('m', 12), ('n', 13), ('o', 14), ('p', 15), ('q', 16), ('r', 17),
            ('s', 18), ('t', 19), ('u', 20), ('v', 21), ('w', 22), ('x', 23),
            ('y', 24), ('z', 25)
        ]
    );

    let mut encoded_password: Vec<u8> = Vec::new();

    for char in input.chars() {
        encoded_password.push(alphabet.get(&char).unwrap().clone());
    }

    return encoded_password;
}


pub fn decode_array(input: &Vec<u8>) -> String {

    let alphabet: HashMap<u8, char> = HashMap::from(
        [
            ( 0, 'a'),( 1, 'b'),( 2, 'c'),( 3, 'd'),( 4, 'e'),( 5, 'f'),
            ( 6, 'g'),( 7, 'h'),( 8, 'i'),( 9, 'j'),(10, 'k'),(11, 'l'),
            (12, 'm'),(13, 'n'),(14, 'o'),(15, 'p'),(16, 'q'),(17, 'r'),
            (18, 's'),(19, 't'),(20, 'u'),(21, 'v'),(22, 'w'),(23, 'x'),
            (24, 'y'),(25, 'z')
        ]
    );

    let mut password = String::new();

    for val in input {
        password.push(alphabet.get(val).unwrap().clone());
    }

    return password
}

fn contains_3_increment(input: &Vec<u8>) -> bool {

    let alphabet = "abcdefghijklmnopqrstuvwxyz";

    for input_index in 0..(input.len()-2) {
        let base_term = &input[input_index..input_index+3];

        for search_index in 0..24 {
            let search_term = Vec::<u8>::from([search_index, search_index + 1, search_index + 2]);
            if &search_term == base_term {
                return true;
            } 
        }
    }

    return false;
}

fn contains_bad_characters(input: &Vec<u8>) -> bool {

    if input.contains(&8) || input.contains(&11) || input.contains(&14) {
        return true;
    }

    return false;
}

fn contains_2_duplicates(input: &Vec<u8>) -> bool {

    let mut num_duplicates = 0;

    for search_index in 0..26 {
        let search_term = &Vec::<u8>::from([search_index, search_index]);
        for input_index in 0..(input.len() - 1) {
            let input_term = &input[input_index..(input_index + 2)];

            if search_term == input_term {
                num_duplicates += 1;
                break;
            }   
        }
    }

    if num_duplicates >= 2 {
        return true;
    }

    return false;
}

pub fn is_good(input: &Vec<u8>) -> bool {

    return !contains_bad_characters(input) && contains_2_duplicates(input) && contains_3_increment(input);

}

pub fn increment(input: &Vec<u8>) -> Vec<u8> {

    let mut output = Vec::<u8>::new();

    let mut inc_next_val = true;
    for input_index in (0..input.len()).rev() {
        if inc_next_val {
            let mut val = &input[input_index] + 1;

            if (val == 8) || (val == 11) || (val == 14) {
                val += 1;
            }

            if val >= 26 {
                output.insert(0, 0);
                inc_next_val = true;
            }
            else {
                output.insert(0, val);
                inc_next_val = false;
            }
        }
        else {
            output.insert(0, input[input_index]);
        }
    }

    return output;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode()
    {
        assert_eq!(encode_string(&String::from("aaaaaaaa")), Vec::<u8>::from([0,0,0,0,0,0,0,0]));
        assert_eq!(encode_string(&String::from("zzzzzzzz")), Vec::<u8>::from([25,25,25,25,25,25,25,25]));
        assert_eq!(encode_string(&String::from("aaaaaaaz")), Vec::<u8>::from([0,0,0,0,0,0,0,25]));
    }

    #[test]
    fn test_decode()
    {
        assert_eq!(decode_array(&Vec::<u8>::from([0,0,0,0,0,0,0,0])), String::from("aaaaaaaa"));
        assert_eq!(decode_array(&Vec::<u8>::from([0,0,0,0,0,0,0,25])), String::from("aaaaaaaz"));
        assert_eq!(decode_array(&Vec::<u8>::from([25,25,25,25,25,25,25,25])), String::from("zzzzzzzz"));
    }

    #[test]
    fn test_contains_3_increment()
    {
        assert_eq!(contains_3_increment(&Vec::<u8>::from([0,0,0,0,0,0,0,0])), false);
        assert_eq!(contains_3_increment(&Vec::<u8>::from([0,1,2,0,0,0,0,0])), true);
        assert_eq!(contains_3_increment(&Vec::<u8>::from([0,23,24,25,0,0,0,0])), true);
    }

    #[test]
    fn test_contains_bad_characters()
    {
        assert_eq!(contains_bad_characters(&Vec::<u8>::from([0,0,0,0,0,0,0,0])), false);
        assert_eq!(contains_bad_characters(&Vec::<u8>::from([0,8,0,0,0,0,0,0])), true);
        assert_eq!(contains_bad_characters(&Vec::<u8>::from([0,0,0,0,11,0,0,0])), true);
        assert_eq!(contains_bad_characters(&Vec::<u8>::from([0,0,14,0,0,0,0,0])), true);
    }

    #[test]
    fn test_contains_duplicates()
    {
        // assert_eq!(contains_2_duplicates(&Vec::<u8>::from([0,0,0,0,0,0,0,0])), false);
        // assert_eq!(contains_2_duplicates(&Vec::<u8>::from([1,1,1,1,1,1,1,1])), false);
        // assert_eq!(contains_2_duplicates(&Vec::<u8>::from([0,0,1,1,0,0,0,0])), true);
        assert_eq!(contains_2_duplicates(&Vec::<u8>::from([0,1,25,25,0,1,1,0])), true);
    }

    #[test]
    fn test_is_good()
    {
        assert_eq!(is_good(&Vec::<u8>::from([0,0,1,1,2,3,0,0])), true);
        assert_eq!(is_good(&Vec::<u8>::from([0,0,1,1,2,3,8,0])), false);
    }

    #[test]
    fn test_increment()
    {
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,0,0,0])), Vec::<u8>::from([0,0,0,0,0,0,0,1]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,0,0,24])), Vec::<u8>::from([0,0,0,0,0,0,0,25]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,0,0,25])), Vec::<u8>::from([0,0,0,0,0,0,1,0]));
        assert_eq!(increment(&Vec::<u8>::from([25,25,25,25,25,25,25,25])), Vec::<u8>::from([0,0,0,0,0,0,0,0]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,7,0,0])), Vec::<u8>::from([0,0,0,0,0,7,0,1]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,7,25,25])), Vec::<u8>::from([0,0,0,0,0,9,0,0]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,0,25,25])), Vec::<u8>::from([0,0,0,0,0,1,0,0]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,10,25,25])), Vec::<u8>::from([0,0,0,0,0,12,0,0]));
        assert_eq!(increment(&Vec::<u8>::from([0,0,0,0,0,13,25,25])), Vec::<u8>::from([0,0,0,0,0,15,0,0]));
    }
}