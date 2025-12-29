use std::fs::File;

fn look_ahead(input: &Vec<u8>) -> Vec<u8> {

    let mut new_string: Vec<u8> = Vec::new();
    let mut temp_buffer = Vec::new();
    temp_buffer.push(input[0]);

    if input.len() == 1 {
        return Vec::from([1, temp_buffer[0]]);
    }

    for index in 1..input.len() {
        let curr_char = input[index];
        if temp_buffer.len() == 0 {
            temp_buffer.push(curr_char);
        }
        else if temp_buffer[0] == curr_char {
            temp_buffer.push(curr_char);
        }
        else {
            new_string.push(temp_buffer.len() as u8);
            new_string.push(temp_buffer[0]);
            temp_buffer = Vec::from([curr_char]);
        }
    }

    new_string.push(temp_buffer.len() as u8);
    new_string.push(temp_buffer[0]);

    return new_string;
}

pub fn part_a(input: &String)
{
    println!("Part B");

    let mut temp: Vec<u8> = input.chars().map(|c| c.to_digit(10).unwrap() as u8).collect();

    for iteration in 0..40 {
        temp = look_ahead(&temp);
        println!("Iteration: {}, len: {}", iteration, temp.len());
    }

    // println!("String: {:?}", temp);
    println!("String len: {}", temp.len());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_look_ahead() {
        assert_eq!(look_ahead(&Vec::from([1])), Vec::from([1,1]));
        assert_eq!(look_ahead(&Vec::from([2])), Vec::from([1,2]));
        assert_eq!(look_ahead(&Vec::from([3])), Vec::from([1,3]));

        assert_eq!(look_ahead(&Vec::from([1, 1])), Vec::from([2, 1]));
        assert_eq!(look_ahead(&Vec::from([1, 2])), Vec::from([1, 1, 1 ,2]));
        assert_eq!(look_ahead(&Vec::from([1, 3])), Vec::from([1, 1, 1, 3]));

        assert_eq!(look_ahead(&Vec::from([2, 1])), Vec::from([1, 2, 1, 1]));
        assert_eq!(look_ahead(&Vec::from([2, 2])), Vec::from([2 ,2]));
        assert_eq!(look_ahead(&Vec::from([2, 3])), Vec::from([1, 2, 1, 3]));

        assert_eq!(look_ahead(&Vec::from([3, 1])), Vec::from([1, 3, 1, 1]));
        assert_eq!(look_ahead(&Vec::from([3, 2])), Vec::from([1, 3, 1 ,2]));
        assert_eq!(look_ahead(&Vec::from([3, 3])), Vec::from([2, 3]));
    }
}