use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn part_a(path: &String)
{
    println!("Part A");

    let file: File = File::open(path).expect("Could not open file");
    let buf_reader:BufReader<File> = BufReader::new(file);

    let file_total = buf_reader.lines().fold(0, |file_acc: i64, line| {
            file_acc + match line {
                Ok(x) => {
                    x.chars().fold(0, |line_acc: i64, b: char| 
                            {
                                line_acc + match b {
                                    '(' => 1,
                                    ')' => -1,
                                    _ => 0,
                                }
                            }
                        )
                },
                Err(_) => 0,
            }
        }
    );

    println!("Total: {}", file_total);
}