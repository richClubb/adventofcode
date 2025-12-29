use std::fs::File;
use std::io::{BufRead, BufReader};

fn get_ribbon(path: &str) -> u64
{
    // 2*l*w + 2*w*h + 2*h*l
    let mut values: Vec<u64> = path.split("x").map(|s| s.parse().unwrap()).collect();
    let length = values[0];
    let width = values[1];
    let height = values[2];

    let volume = length * width * height;

    values.sort();
    
    let perimeter = (values[0] * 2) + (values[1] * 2);

    return volume + perimeter;
}

pub fn part_b(path: &String)
{
    println!("Part B");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let mut total = 0;
    for line in buf_reader.lines()
    {
        total = total + get_ribbon(&line.unwrap());
    }

    println!("Total {}", total);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ribbon() {
        assert_eq!(get_ribbon("2x3x4"), 34);
        assert_eq!(get_ribbon("1x1x10"), 14);
    }
}