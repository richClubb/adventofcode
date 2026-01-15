use std::fs::File;
use std::io::{BufRead, BufReader};

fn get_area(path: &str) -> u64
{
    // 2*l*w + 2*w*h + 2*h*l
    let values: Vec<u64> = path.split("x").map(|s| s.parse().unwrap()).collect();
    let length = values[0];
    let width = values[1];
    let height = values[2];

    let side1 = length * width ;
    let side2 = width * height ;
    let side3 = height * length;

    let sides = Vec::from([side1, side2, side3]);
    let min_side = sides.iter().min().unwrap();
    
    return (side1 * 2) + (side2 * 2) + (side3 * 2) + min_side;
}

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let total = buf_reader.lines().fold(0, |acc, line| {
            acc + get_area(&line.unwrap())
        }
    );

    println!("Total {}", total);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_area() {
        assert_eq!(get_area("2x3x4"), 58);
        assert_eq!(get_area("1x1x10"), 43);
    }
}