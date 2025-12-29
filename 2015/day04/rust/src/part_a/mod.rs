use md5::{Md5, Digest};

fn generate_hash(key: String) -> bool{

    let result = Md5::digest(key);

    if (result[0] == 0) && (result[1] == 0) && (result[2] < 15) {
        println!("result {:?}", result);
        return true;
    }
    
    return false
}


pub fn part_a(key: &String)
{
    println!("Part A");

    let mut number = 0;
    loop {

        let current_str = key.to_owned() + &number.to_string();
        if generate_hash(current_str) {
            println!("{}", number);
            break;
        }

        number += 1;
    }
    
}
