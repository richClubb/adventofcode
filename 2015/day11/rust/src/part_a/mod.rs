use crate::encode_decode::{decode_array, encode_string, increment, is_good};


pub fn part_a(input: &String)
{
    println!("Part A");

    let mut curr_password = encode_string(input);
    let mut final_password: String;
    
    loop {
        let new_password = increment(&curr_password);

        if is_good(&new_password) {
            final_password = decode_array(&new_password);
            break;
        }

        curr_password = new_password;
    }

    println!("New password is: {}", final_password);

}

#[cfg(test)]
mod tests {
    use super::*;

}